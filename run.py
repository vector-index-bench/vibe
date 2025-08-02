import multiprocessing

from vibe.main import main

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    main()
