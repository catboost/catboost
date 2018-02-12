var moduleBase = 'jupyter-js-widgets';

try{
    moduleBase = Jupyter.WidgetManager.prototype.loadClass.toString().indexOf('@jupyter-widgets/base') > -1 ? '@jupyter-widgets/base' : 'jupyter-js-widgets';
} catch(e) {}

define('catboost_module', [moduleBase], function(widgets) {
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
                    this.process_row(parent, data[path], {name: 'learn_error', type: 'learn'});
                    this.process_row(parent, data[path], {name: 'test_error', type: 'test'});
                }
            }
        },

        process_row: function(parent, data, params) {
            var catboostIpython = getInstance(parent),
                path = data.path,
                items = data.content.rows[params.name],
                chunks = [],
                chunkItems,
                firstIndex = 1;

            if (!items.length) {
                return;
            }

            if (!catboostIpython.lastIndexes[path]) {
                catboostIpython.lastIndexes[path] = {};
            }

            // file can contains unfinished lines
            // so we need to read it again to fix incorrect displayed data
            /*
            if (!catboostIpython.lastIndexes[path][params.type]) {
                firstIndex = 1;
            } else {
                firstIndex = Math.max(catboostIpython.lastIndexes[path][params.type] - 1, 1);
            }
            */

            catboostIpython.lastIndexes[path][params.type] = items.length;

            for (var i = firstIndex; i < items.length; i++) {
                chunkItems = items[i].map(function(item) {
                    return Number(item);
                });

                chunks.push(chunkItems);
            }

            catboostIpython.addMeta(data.path, data.content.rows['meta']);

            catboostIpython.setTime(data.path, data.content.rows['time_left']);

            catboostIpython.addPoints(parent, {
                chunks: chunks,
                path: data.path,
                train: data.name,
                fields: items[0]
            }, params.type);
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
