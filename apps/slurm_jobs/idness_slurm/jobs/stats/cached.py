from src.apis.rw import IODict

cached_files = [
    '../../cached_models.cs'
]

for file in cached_files:
    cs = IODict(file)
    cs.load()
    print(cs.cached.keys())
