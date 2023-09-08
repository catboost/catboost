#include "typedesc.h"

#include <util/generic/singleton.h>

static void RegisterJSONBridgeImpl() {
    PyRun_SimpleString("import json\n"
                       "class PyBindEncoder(json.JSONEncoder):\n"
                       "    def default(self, obj):\n"
                       "        if isinstance(obj, bytes):\n"
                       "            try:\n"
                       "                return obj.decode()\n"
                       "            except UnicodeDecodeError:\n"
                       "                return obj.hex()\n"
                       "        dct = None\n"
                       "        if hasattr(obj, '__getstate__'):\n"
                       "            dct = obj.__getstate__()\n"
                       "        elif hasattr(obj, '__dict__'):\n"
                       "            dct = obj.__dict__\n"
                       "        if dct is None:\n"
                       "            return json.JSONEncoder.default(self, obj)\n"
                       "        if hasattr(obj, '__class__'):\n"
                       "            if hasattr(obj.__class__, '__name__'):\n"
                       "                dct['__name__'] = obj.__class__.__name__\n"
                       "            if hasattr(obj.__class__, '__module__'):\n"
                       "                dct['__module__'] = obj.__class__.__module__\n"
                       "        if hasattr(obj, 'GetPropertiesNames'):\n"
                       "            dct['__properties__'] = obj.GetPropertiesNames()\n"
                       "        return dct");

    PyRun_SimpleString("def PyBindObjectHook(dct):\n"
                       "    if '__name__' in dct:\n"
                       "        name = dct['__name__']\n"
                       "        module = dct['__module__']\n"
                       "        del dct['__name__']\n"
                       "        del dct['__module__']\n"
                       "        cls = getattr(__import__(module), name)\n"
                       "        if '__properties__' in dct:\n"
                       "            props = dct['__properties__']\n"
                       "            del dct['__properties__']\n"
                       "            if len(props) == 0:\n"
                       "                return dct\n"
                       "            instance = cls(__properties__ = props)\n"
                       "        else:\n"
                       "            instance = cls()\n"
                       "        if hasattr(instance, '__setstate__'):\n"
                       "            instance.__setstate__(dct)\n"
                       "        elif hasattr(instance, '__dict__'):\n"
                       "            instance.__dict__ = dct\n"
                       "        else:\n"
                       "            return dct\n"
                       "        return instance\n"
                       "    return dct");

    PyRun_SimpleString("def json_dump(*args, **kwargs):\n"
                       "    kwargs['cls'] = PyBindEncoder\n"
                       "    return json.dump(*args, **kwargs)\n"
                       "def json_dumps(*args, **kwargs):\n"
                       "    kwargs['cls'] = PyBindEncoder\n"
                       "    return json.dumps(*args, **kwargs)");

    PyRun_SimpleString("def json_load(*args, **kwargs):\n"
                       "    kwargs['object_hook'] = PyBindObjectHook\n"
                       "    return json.load(*args, **kwargs)\n"
                       "def json_loads(*args, **kwargs):\n"
                       "    kwargs['object_hook'] = PyBindObjectHook\n"
                       "    return json.loads(*args, **kwargs)");
}

namespace {
    struct TJSONBridge {
        TJSONBridge() {
            RegisterJSONBridgeImpl();
        }
    };
}

void NPyBind::RegisterJSONBridge() {
    Singleton<TJSONBridge>();
}
