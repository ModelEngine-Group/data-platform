import os
if hasattr(os, "add_dll_directory"):
    # specify your own openslide binary locations
    with os.add_dll_directory(os.path.join(os.getcwd(), 'bin')):
        import openslide
else:
    # os.environ['PATH'] = 'C:\\research\\openslide\\bin' + ';' + os.environ['PATH']
    # #can either specify openslide bin path in PATH, or add it dynamically
    import openslide
