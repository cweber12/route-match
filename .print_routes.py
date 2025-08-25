import sys, os
sys.path.insert(0, os.path.abspath('.'))
import importlib
m = importlib.import_module('app.main')
print('Loaded app.main')
print([r.path for r in m.app.routes])
