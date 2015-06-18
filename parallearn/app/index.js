var pythonjs = require('python-js');
var pycode = "a = []; a.append('hello'); a.append('world'); print(a)";
var jscode = pythonjs.translator.to_javascript( pycode );
eval( pythonjs.runtime.javascript + jscode );
