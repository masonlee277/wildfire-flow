import sys
import os 
import importlib


def reload_mods():
    # Import your custom modules
    import src.data.preprocessor
    import src.models.normalizing_flow
    import src.models.cnf_utils

    # Reload the modules
    importlib.reload(src.data.preprocessor)
    importlib.reload(src.models.normalizing_flow)
    importlib.reload(src.models.cnf_utils)
    
    print("Modules reloaded successfully!")
