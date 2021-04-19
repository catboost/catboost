var plugin = require('./index');
var base = require('@jupyter-widgets/base');

module.exports = {
    id: 'catboost-widget:plugin',
    requires: [base.IJupyterWidgetRegistry],
    activate: function (app, widgets) {
        widgets.registerWidget({
            name: 'catboost-widget',
            version: plugin.version,
            exports: plugin
        });
    },
    autoStart: true
};

