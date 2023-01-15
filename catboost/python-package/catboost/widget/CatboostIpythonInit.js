var debug = false;

if (debug) {
    require.config({
        shim:{
            "custom/CatboostIpythonPlotly":{
                deps:["custom/plotly-basic.min"]
            }
        }
    })

    require.undef('catboost_module');
    require.undef('custom/CatboostIpythonPlotly');
}

var moduleBase = '@jupyter-widgets/base';
var modules = [moduleBase];

if (debug) {
    modules.push('custom/CatboostIpythonPlotly');
}

define('catboost_module', modules, function(widgets) {
    var getInstance = function(el) {
            var id = $(el).attr('catboost-id');

            if (!id) {
                return null;
            }

            id = id.replace('catboost_', '');

            if (!window.catboostIpythonInstances[id]) {
                return null;
            }

            return window.catboostIpythonInstances[id];
        },
        addInstance = function(el) {
            $(el).attr('catboost-id', 'catboost_' + window.catboostIpythonIndex);

            var catboostIpython = new CatboostIpython();
            catboostIpython.index = window.catboostIpythonIndex;
            catboostIpython.plotly = window.Plotly;
            if (debug) {
                catboostIpython.loadStyles('/custom/CatboostIpython.css', function(){catboostIpython.resizeCharts();})
            }

            window.catboostIpythonInstances[window.catboostIpythonIndex] = catboostIpython;

            window.catboostIpythonIndex++;

            return catboostIpython;
        };

    var CatboostIpythonWidget = widgets.DOMWidgetView.extend({

        initialize: function() {
            CatboostIpythonWidget.__super__.initialize.apply(this, arguments);

            if (!window.catboostIpythonInstances) {
                window.catboostIpythonInstances = {};
            }

            if (typeof window.catboostIpythonIndex === 'undefined') {
                window.catboostIpythonIndex = 0;
            }

            var catboostIpythonInstance = getInstance(this.el);

            if (!catboostIpythonInstance) {
                catboostIpythonInstance = addInstance(this.el);
            }

            catboostIpythonInstance.init();
        },

        render: function() {
            this.value_changed();
            this.model.on('change:value', this.value_changed, this);
        },

        update: function() {
            this.value_changed();
        },

        value_changed: function() {
            this.el.style['width'] = this.model.get('width');
            this.el.style['height'] = this.model.get('height');
            this.displayed.then(_.bind(this.render_charts, this));
        },

        process_all: function(parent, params) {
            var data = params.data;

            for (var path in data) {
                if (data.hasOwnProperty(path)) {
                    this.process_row(parent, data[path])
                }
            }
        },

        process_row: function(parent, data) {
            var catboostIpython = getInstance(parent),
                path = data.path,
                content = data.content,
                items = content.data.iterations,
                firstIndex = 0,
                chunks = [];

            if (!items || !items.length) {
                return;
            }

            if (!catboostIpython.lastIndex) {
                catboostIpython.lastIndex = {}
            }

            if (catboostIpython.lastIndex[path]) {
                firstIndex = catboostIpython.lastIndex[path] + 1;
            }

            catboostIpython.lastIndex[path] = items.length - 1;

            for (var i = firstIndex; i < items.length; i++) {
                chunks.push(items[i]);
            }

            catboostIpython.addMeta(data.path, content.data.meta);

            catboostIpython.addPoints(parent, {
                chunks: chunks,
                train: data.name,
                path: data.path
            });
        },

        render_charts: function () {
            this.process_all(this.el, {
                data: this.model.get('data')
            });

            return this;
        }
    });

    return {
        CatboostIpythonWidgetView: CatboostIpythonWidget
    };
});
