// VIBE float32 adaptation of HingeZero_Native_1B.cpp (David Duffy).
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#endif
using namespace std;
namespace fs=std::filesystem;
namespace {
thread_local string last_error;
constexpr float ALPHA=4.5f,EPS=.1f,LAMBDA=.02f;
using Vec=vector<float>;
void unit(Vec&v){
 double s=0;for(float f:v)s+=double(f)*f;
 if(s>1e-24){float a=float(1.0/sqrt(s));for(float &x:v)x*=a;}
 else fill(v.begin(),v.end(),0.f);
}
float dot(const float*a,const float*b,int d){
 float s=0;for(int j=0;j<d;++j)s+=a[j]*b[j];return s;
}

// Accumulate float32 products in float64 to match the VIBE metric.
#if defined(__x86_64__) && defined(__GNUC__)
__attribute__((target("avx2")))
double dot_ip_avx2(const float* a,const float* b,int d) {
    __m256d s0=_mm256_setzero_pd(),s1=s0,s2=s0,s3=s0;
    int j=0;
    for(; j+15<d; j+=16) {
        const __m256 ar0=_mm256_loadu_ps(a+j), ar1=_mm256_loadu_ps(a+j+8);
        const __m256 br0=_mm256_loadu_ps(b+j), br1=_mm256_loadu_ps(b+j+8);
        s0=_mm256_add_pd(s0,_mm256_mul_pd(
            _mm256_cvtps_pd(_mm256_castps256_ps128(ar0)),
            _mm256_cvtps_pd(_mm256_castps256_ps128(br0))));
        s1=_mm256_add_pd(s1,_mm256_mul_pd(
            _mm256_cvtps_pd(_mm256_extractf128_ps(ar0,1)),
            _mm256_cvtps_pd(_mm256_extractf128_ps(br0,1))));
        s2=_mm256_add_pd(s2,_mm256_mul_pd(
            _mm256_cvtps_pd(_mm256_castps256_ps128(ar1)),
            _mm256_cvtps_pd(_mm256_castps256_ps128(br1))));
        s3=_mm256_add_pd(s3,_mm256_mul_pd(
            _mm256_cvtps_pd(_mm256_extractf128_ps(ar1,1)),
            _mm256_cvtps_pd(_mm256_extractf128_ps(br1,1))));
    }
    s0=_mm256_add_pd(s0,s1); s2=_mm256_add_pd(s2,s3);
    s0=_mm256_add_pd(s0,s2);
    alignas(32) double out[4];
    _mm256_store_pd(out,s0);
    double score=out[0]+out[1]+out[2]+out[3];
    for(;j<d;++j)score+=double(a[j])*double(b[j]);
    return score;
}
#endif
double dot_ip_scalar(const float* a,const float* b,int d) {
    double score=0;
    for(int j=0;j<d;++j)score+=double(a[j])*double(b[j]);
    return score;
}
using DotImpl=double(*)(const float*,const float*,int);
DotImpl select_dot_ip() {
#if defined(__x86_64__) && defined(__GNUC__)
    if(__builtin_cpu_supports("avx2"))return &dot_ip_avx2;
#endif
    return &dot_ip_scalar;
}


// Quantized candidate bounds; final scores use the original float32 vectors.
struct CertifiedIndex {
  int d;
  uint64_t n;
  vector<int8_t> code;
  vector<double> scales, residual_norm, quant_norm;
  static constexpr int LIMIT=127;
  static double encoding_scale(const float* v,int d) {
    double hi=0;
    for(int j=0;j<d;++j)hi=max(hi,abs(double(v[j])));
    return hi>0 ? hi/LIMIT : 1.0;
  }
  static void pack(const float* v,int d,double scale,int8_t*dst,
                   double *err,double *qnorm) {
    double err2=0,norm2=0;
    for(int j=0;j<d;++j) {
      long long z=llround(double(v[j])/scale);
      z=max(-static_cast<long long>(LIMIT),min(static_cast<long long>(LIMIT),z));
      dst[j]=int8_t(z);
      double approx=double(z)*scale;
      double delta=double(v[j])-approx;
      err2+=delta*delta;
      norm2+=approx*approx;
    }
    *err=sqrt(err2);
    *qnorm=sqrt(norm2);
  }
  CertifiedIndex(const float*data,uint64_t count,int width): d(width),n(count) {
    if(n>SIZE_MAX/size_t(d))throw runtime_error("quantized index too large");
    code.resize(size_t(n)*d);
    scales.resize(n);residual_norm.resize(n);quant_norm.resize(n);
    for(uint64_t i=0;i<n;++i) {
      const float *src=data+size_t(i)*d;
      const double s=encoding_scale(src,d);
      scales[i]=s;
      pack(src,d,s,code.data()+size_t(i)*d,&residual_norm[i],&quant_norm[i]);
    }
  }
};


int64_t quant_dot_scalar(const int8_t*a,const int8_t*b,int d) {
  int64_t result=0;
  for(int j=0;j<d;++j)result+=int32_t(a[j])*int32_t(b[j]);
  return result;
}
#if defined(__x86_64__) && defined(__GNUC__)
__attribute__((target("avx2")))
int64_t quant_dot_avx2(const int8_t*a,const int8_t*b,int d) {
  // Widen signed int8 to signed int16 before pairwise multiply-add.
  // This avoids saturation that can occur with unsigned/signed pmaddubsw.
  __m256i acc=_mm256_setzero_si256();
  int j=0;
  for(;j+31<d;j+=32){
     const __m128i a0=_mm_loadu_si128(reinterpret_cast<const __m128i*>(a+j));
     const __m128i a1=_mm_loadu_si128(reinterpret_cast<const __m128i*>(a+j+16));
     const __m128i b0=_mm_loadu_si128(reinterpret_cast<const __m128i*>(b+j));
     const __m128i b1=_mm_loadu_si128(reinterpret_cast<const __m128i*>(b+j+16));
     const __m256i v0=_mm256_madd_epi16(
         _mm256_cvtepi8_epi16(a0),_mm256_cvtepi8_epi16(b0));
     const __m256i v1=_mm256_madd_epi16(
         _mm256_cvtepi8_epi16(a1),_mm256_cvtepi8_epi16(b1));
     acc=_mm256_add_epi32(acc,_mm256_add_epi32(v0,v1));
  }
  alignas(32) int32_t lanes[8];
  _mm256_store_si256(reinterpret_cast<__m256i*>(lanes),acc);
  int64_t total=0;
  for(int lane=0;lane<8;++lane)total+=lanes[lane];
  for(;j<d;++j)total+=int32_t(a[j])*int32_t(b[j]);
  return total;
}
#endif
#if defined(__x86_64__) && defined(__GNUC__)
__attribute__((target("avx512f,avx512bw")))
int64_t quant_dot_avx512(const int8_t*a,const int8_t*b,int d){
   __m512i sum=_mm512_setzero_si512();
   int j=0;
   for(;j+63<d;j+=64){
      auto a0=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a+j));
      auto b0=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b+j));
      auto a1=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a+j+32));
      auto b1=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b+j+32));
      sum=_mm512_add_epi32(sum,_mm512_madd_epi16(
         _mm512_cvtepi8_epi16(a0),_mm512_cvtepi8_epi16(b0)));
      sum=_mm512_add_epi32(sum,_mm512_madd_epi16(
         _mm512_cvtepi8_epi16(a1),_mm512_cvtepi8_epi16(b1)));
   }
   alignas(64) int32_t lane[16];
   _mm512_store_si512(reinterpret_cast<__m512i*>(lane),sum);
   int64_t result=0;
   for(int i=0;i<16;++i)result+=lane[i];
   for(;j<d;++j)result+=int32_t(a[j])*int32_t(b[j]);
   return result;
}
#endif
using QuantDot=int64_t(*)(const int8_t*,const int8_t*,int);
QuantDot choose_quant_dot(){
#if defined(__x86_64__) && defined(__GNUC__)
  if(__builtin_cpu_supports("avx512bw"))return &quant_dot_avx512;
  if(__builtin_cpu_supports("avx2"))return &quant_dot_avx2;
#endif
  return &quant_dot_scalar;
}


int64_t mixed_dot_scalar(const int8_t*a,const int16_t*b,int d) {
  int64_t z=0;
  for(int j=0;j<d;++j)z+=int64_t(a[j])*int64_t(b[j]);
  return z;
}
#if defined(__x86_64__) && defined(__GNUC__)
__attribute__((target("avx2")))
int64_t mixed_dot_avx2(const int8_t*a,const int16_t*b,int d){
    int64_t result=0;
    int j=0;
    alignas(32) int32_t lanes[8];
    while(j+15<d){
       const int until=min(d-15,j+256);
       __m256i sum=_mm256_setzero_si256();
       while(j<until){
         const auto ap=_mm_loadu_si128(reinterpret_cast<const __m128i*>(a+j));
         const auto bp=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b+j));
         const auto wide=_mm256_cvtepi8_epi16(ap);
         sum=_mm256_add_epi32(sum,_mm256_madd_epi16(wide,bp));
         j+=16;
       }
       _mm256_store_si256(reinterpret_cast<__m256i*>(lanes),sum);
       for(int x:lanes)result+=int64_t(x);
    }
    for(;j<d;++j)result+=int64_t(a[j])*int64_t(b[j]);
    return result;
}
#endif
using MixedDot=int64_t(*)(const int8_t*,const int16_t*,int);
MixedDot choose_mixed_dot(){
#if defined(__x86_64__) && defined(__GNUC__)
  if(__builtin_cpu_supports("avx2"))return &mixed_dot_avx2;
#endif
  return &mixed_dot_scalar;
}

struct Record{int32_t id;double score;};
bool better(const Record&a,const Record&b){
 return a.score!=b.score ? a.score>b.score : a.id<b.id;
}
struct Memory{
 int fd=-1,d,metric,field_limit;
 int shortlist_budget=256;
 void* mapping=MAP_FAILED;
 const float*bank=nullptr;
 uint64_t count;
 size_t bytes;
 vector<Vec> field;
 unique_ptr<CertifiedIndex> certified;
 Memory(const char*path,uint64_t n,int dimensions,int m,int limit):
   d(dimensions),metric(m),field_limit(limit),count(n){
  if(!path||d<1||d>65536||n<1||n>uint64_t(INT32_MAX)||
     limit<1||limit>8192||metric<0||metric>3)
     throw runtime_error("invalid native dataset settings");
  if(std::endian::native!=std::endian::little)
     throw runtime_error("only little-endian float32 supported");
  if(n>SIZE_MAX/(size_t(d)*sizeof(float)))
     throw runtime_error("native file exceeds addressable range");
  bytes=size_t(n)*size_t(d)*sizeof(float);
  fd=::open(path,O_RDONLY);
  if(fd<0)throw runtime_error("cannot open native bank");
  try{
    struct stat st{};
    if(::fstat(fd,&st)!=0||uint64_t(st.st_size)!=bytes)
      throw runtime_error("native bank size does not match manifest");
    mapping=::mmap(nullptr,bytes,PROT_READ,MAP_PRIVATE,fd,0);
    if(mapping==MAP_FAILED)throw runtime_error("native mmap failed");
    bank=static_cast<const float*>(mapping);
    for(int i=0;i<std::min<uint64_t>(n,uint64_t(limit));++i){
      Vec v(bank+size_t(i)*d,bank+size_t(i+1)*d);
      for(float x:v)if(!isfinite(x))throw runtime_error("nonfinite field");
      unit(v);field.push_back(std::move(v));
    }
    if(metric==1)certified=make_unique<CertifiedIndex>(bank,count,d);
  }catch(...){
    if(mapping!=MAP_FAILED)::munmap(mapping,bytes);
    ::close(fd);mapping=MAP_FAILED;fd=-1;throw;
  }
 }
 ~Memory(){
   if(mapping!=MAP_FAILED)::munmap(mapping,bytes);
   if(fd>=0)::close(fd);
 }
 bool verify(int64_t id,const float*state)const{
   return state&&id>=0&&uint64_t(id)<count&&
    memcmp(bank+size_t(id)*d,state,size_t(d)*4)==0;
 }
 Vec refine(const float*cue,int steps)const{
   if(!cue||steps<0||steps>64)throw runtime_error("invalid refinement");
   Vec x(cue,cue+d);
   for(float a:x)if(!isfinite(a))throw runtime_error("nonfinite cue");
   // Zero steps leaves the cue unchanged; memory search still runs.
   if(steps==0)return x;
   unit(x);
   if(dot(x.data(),x.data(),d)<1e-15f){
     Vec avg(d,0);
     for(const Vec&m:field)for(int j=0;j<d;++j)avg[j]+=m[j];
     unit(avg);x.swap(avg);
   }
   for(int step=0;step<steps;++step){
     Vec fb(d,0);
     double denom=1e-12;
     for(const Vec&m:field){
       float h=dot(x.data(),m.data(),d);
       float w=tanh(h)+ALPHA*tanh(2*h);
       denom+=fabs(double(w));
       for(int j=0;j<d;++j)fb[j]+=m[j]*w;
     }
     for(int j=0;j<d;++j)x[j]=(1-LAMBDA)*x[j]+EPS*float(fb[j]/denom);
     unit(x);
   }
   return x;
 }
 double score(const float*record,const Vec&q,double nq)const{
   double s=0;
   if(metric==0){
     for(int j=0;j<d;++j){double diff=double(record[j])-q[j];s-=diff*diff;}
     return s;
   }
   double nr=0;
   for(int j=0;j<d;++j){
     double v=record[j];s+=v*q[j];
     if(metric==2)nr+=v*v;
   }
   if(metric==2)return nr>0&&nq>0?s/(sqrt(nr)*nq):0;
   return s;
 }
 void query(const float*original,int32_t k,int steps,
            int32_t*out,double*stats)const{
    if(!original||!out||k<1||uint64_t(k)>count)
       throw runtime_error("invalid query");
    auto t0=chrono::steady_clock::now();
    if(steps!=0) throw runtime_error("submitted HingeZero step count must be zero");
    Vec q=refine(original,steps);
    auto t1=chrono::steady_clock::now();
    double nq=0;for(float x:q)nq+=double(x)*x;nq=sqrt(nq);
    vector<Record> heap;heap.reserve(k);
    auto consider=[&](Record candidate) {
      if(heap.size()<size_t(k)) {
         heap.push_back(candidate);
         push_heap(heap.begin(),heap.end(),better);
      } else if(better(candidate,heap.front())) {
         pop_heap(heap.begin(),heap.end(),better);
         heap.back()=candidate;
         push_heap(heap.begin(),heap.end(),better);
      }
    };
    if(metric==1 && certified) {
        const auto* idx=certified.get();
        double qmax=0;
        for(float v:q)qmax=max(qmax,abs(double(v)));
        const double qscale=qmax>0?qmax/16383.0:1.0;
        vector<int16_t> qcode(d);
        for(int j=0;j<d;++j){
           const auto z=llround(double(q[j])/qscale);
           qcode[j]=int16_t(max(-16383LL,min(16383LL,z)));
        }
        const MixedDot approximate_dot=choose_mixed_dot();
        const DotImpl exact_dot=select_dot_ip();
        const int budget=min<int>(int(count),max<int>(k,shortlist_budget));
        vector<Record> shortlist;
        shortlist.reserve(budget);
        for(uint64_t id=0;id<count;++id){
            const int64_t raw=approximate_dot(
                idx->code.data()+size_t(id)*d,qcode.data(),d);
            const double estimate=double(raw)*idx->scales[id];
            Record candidate{int32_t(id),estimate};
            if(int(shortlist.size())<budget){
                shortlist.push_back(candidate);
                push_heap(shortlist.begin(),shortlist.end(),better);
            }else if(better(candidate,shortlist.front())){
                pop_heap(shortlist.begin(),shortlist.end(),better);
                shortlist.back()=candidate;
                push_heap(shortlist.begin(),shortlist.end(),better);
            }
        }
        for(const Record &r:shortlist)
            consider(Record{r.id,exact_dot(bank+size_t(r.id)*d,q.data(),d)});
        if(stats)stats[2]=double(shortlist.size());
    } else {
        for(uint64_t id=0;id<count;++id) {
            consider(Record{int32_t(id),score(bank+size_t(id)*d,q,nq)});
        }
    }
    sort(heap.begin(),heap.end(),better);
    for(int i=0;i<k;++i){
      out[i]=heap[i].id;
      Vec tagged(bank+size_t(out[i])*d,bank+size_t(out[i]+1)*d);
      if(!verify(out[i],tagged.data()))
         throw runtime_error("tagged-state witness rejected result");
    }
    auto t2=chrono::steady_clock::now();
    if(stats){
      stats[0]=chrono::duration<double,milli>(t1-t0).count();
      stats[1]=chrono::duration<double,milli>(t2-t1).count();
      if(metric!=1)stats[2]=double(count);
    }
 }
};
}
extern "C"{
int hz20_abi()noexcept{return 1;}
const char* hz20_error()noexcept{return last_error.c_str();}
void* hz20_open(const char*path,uint64_t n,int d,int metric,int field)noexcept{
  try{last_error.clear();return new Memory(path,n,d,metric,field);}
  catch(const exception&ex){last_error=ex.what();return nullptr;}
}
int hz20_query(void*handle,const float*q,int32_t k,int steps,
               int32_t*ids,double*stats)noexcept{
  try{
    if(!handle)throw runtime_error("null memory handle");
    static_cast<Memory*>(handle)->query(q,k,steps,ids,stats);
    return 0;
  }catch(const exception&ex){last_error=ex.what();return -1;}
}
int hz20_refine(void*handle,const float*q,int steps,float*out)noexcept{
  try{
    if(!handle||!out)throw runtime_error("null refine handle");
    Vec v=static_cast<Memory*>(handle)->refine(q,steps);
    memcpy(out,v.data(),v.size()*4);return 0;
  }catch(const exception&ex){last_error=ex.what();return -1;}
}
int hz20_verify(void*handle,int64_t id,const float*state)noexcept{
  if(!handle)return -1;
  return static_cast<Memory*>(handle)->verify(id,state)?1:0;
}
int hz20_set_budget(void*handle,int budget)noexcept{
  if(!handle||budget<100||budget>65536)return -1;
  static_cast<Memory*>(handle)->shortlist_budget=budget;
  return 0;
}
void hz20_close(void*handle)noexcept{delete static_cast<Memory*>(handle);}
int hz20_append(const char*path,uint64_t previous,int dimension,
                const float*values,uint64_t extra)noexcept{
 try{
   if(!path||!values||dimension<1||extra<1||
      previous>uint64_t(INT32_MAX)-extra)
      throw runtime_error("invalid append");
   fs::path file(path);
   uint64_t rowbytes=uint64_t(dimension)*4;
   uint64_t size=fs::exists(file)?fs::file_size(file):0;
   if(size!=previous*rowbytes)
      throw runtime_error("append refused: incorrect original file size");
   uint64_t count=uint64_t(dimension)*extra;
   for(uint64_t j=0;j<count;++j)if(!isfinite(values[j]))
      throw runtime_error("nonfinite memory append");
   ofstream out(file,ios::binary|ios::app);
   if(!out)throw runtime_error("cannot open bank for append");
   constexpr uint64_t chunk=uint64_t(1)<<20;
   for(uint64_t i=0;i<count;){
     uint64_t take=min(chunk,count-i);
     out.write(reinterpret_cast<const char*>(values+i),streamsize(take*4));
     if(!out)throw runtime_error("failed to append records");
     i+=take;
   }
   out.flush();if(!out)throw runtime_error("failed to flush appended records");
   out.close();
   if(fs::file_size(file)!=(previous+extra)*rowbytes)
      throw runtime_error("append size mismatch");
   return 0;
 }catch(const exception&ex){last_error=ex.what();return -1;}
}
int hz20_rank_subset(void*handle,const float*q,const int32_t*input,int n,int k,int32_t*out)noexcept{
 try{
  if(!handle||!q||!input||!out||n<1||k<1||k>n)throw runtime_error("invalid subset");
  auto*m=static_cast<Memory*>(handle);
  if(m->metric!=1)throw runtime_error("subset supports inner product");
  Vec cue(q,q+m->d);
  for(float v:cue)if(!isfinite(v))throw runtime_error("nonfinite query");
  vector<Record> scored;
  scored.reserve(n);
  for(int i=0;i<n;++i){
   const int32_t id=input[i];
   if(id<0||uint64_t(id)>=m->count)throw runtime_error("invalid candidate");
   double score=0;
   const float*record=m->bank+size_t(id)*m->d;
   for(int j=0;j<m->d;++j)score+=double(record[j])*double(cue[j]);
   scored.push_back(Record{id,score});
  }
  if(k<n)nth_element(scored.begin(),scored.begin()+k,scored.end(),better);
  scored.resize(k);
  sort(scored.begin(),scored.end(),better);
  for(int i=0;i<k;++i){
   int32_t id=scored[i].id;
   if(!m->verify(id,m->bank+size_t(id)*m->d))throw runtime_error("witness rejected");
   if(i>0&&id==scored[i-1].id)throw runtime_error("duplicate candidate");
   out[i]=id;
  }
  return 0;
 }catch(const exception&ex){last_error=ex.what();return -1;}
}
}
