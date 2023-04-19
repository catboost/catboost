require('./CatboostIpython.css');

var widgets = require('@jupyter-widgets/base');
var _ = require('lodash');
var Plotly = require('plotly.js-dist-min');
var $ = require('jquery');
var widget_version = require('../package.json').version;

/* global Highcharts, Plotly*/

/* eslint-disable */
function throttle(fn, timeout, invokeAsap, ctx) {
    var typeofInvokeAsap = typeof invokeAsap;
    if (typeofInvokeAsap === 'undefined') {
        invokeAsap = true;
    } else if (arguments.length === 3 && typeofInvokeAsap !== 'boolean') {
        ctx = invokeAsap;
        invokeAsap = true;
    }

    var timer, args, needInvoke,
        wrapper = function () {
            if (needInvoke) {
                fn.apply(ctx, args);
                needInvoke = false;
                timer = setTimeout(wrapper, timeout);
            } else {
                timer = null;
            }
        };

    return function () {
        args = arguments;
        ctx || (ctx = this);
        needInvoke = true;

        if (!timer) {
            invokeAsap ?
                wrapper() :
                timer = setTimeout(wrapper, timeout);
        }
    };
}

/* eslint-enable */
function CatboostIpython() {
}

CatboostIpython.prototype.init = function () {
    this.charts = {};
    /*
        {
            "rmse_1": {
                ...
            }
        }
    */
    this.traces = {};
    /*
        {
            "rmse_1": {
                name: "rmse",
                id: "rmse_1",
                parent: "div",
                traces: [
                    {
                        name: "current;learn;0;;;",
                        x: [],
                        y: []
                    },
                    {
                        name: "current;learn;0;smoothed;;",
                        x: [],
                        y: []
                    },
                    {
                        name: "current;learn;1;;;",
                        x: [],
                        y: []
                    },
                    {
                        name: "current;learn;1;smoothed;;",
                        x: [],
                        y: []
                    },
                    {
                        name: "current;test;0;;;",
                        x: [],
                        y: []
                    },
                    {
                        name: "current;test;0;smoothed;;",
                        x: [],
                        y: []
                    },
                    {
                        name: "current;test;0;;best_point;",
                        x: [],
                        y: []
                    },
                    {
                        name: "current;test;0;;;best_value",
                        x: [],
                        y: []
                    }
                ]
            }
        }
    */

    this.hovertextParameters = [];
    this.chartsToRedraw = {};
    this.lastIndexes = {};
    this.smoothness = -1;
    this.layoutDisabled = {
        series: {},
        traces: {}
    };
    this.clickMode = false;
    this.logarithmMode = 'linear';
    this.lastSmooth = 0;
    this.layout = null;
    this.activeTab = '';
    this.meta = {};
    this.timeLeft = {};

    this.hasCVMode = false;
    this.stddevEnabled = false;

    this.colors = [
        '#68E256',
        '#56AEE2',
        '#CF56E2',
        '#E28956',
        '#56E289',
        '#5668E2',
        '#E256AE',
        '#E2CF56',
        '#56E2CF',
        '#8A56E2',
        '#E25668',
        '#AEE256'
    ];
    this.colorsByPath = {};
    this.colorIndex = 0;
    this.lossFuncs = {};

    this.isCVinited = false;
};

/* eslint-disable */
CatboostIpython.prototype.loadStyles = function (path, fn, scope) {
    $('link[catboost="1"]').remove();

    var head = document.getElementsByTagName('head')[0], // reference to document.head for appending/ removing link nodes
        link = document.createElement('link');           // create the link node
    link.setAttribute('href', path);
    link.setAttribute('rel', 'stylesheet');
    link.setAttribute('type', 'text/css');
    link.setAttribute('catboost', '1');

    var sheet, cssRules;
    // get the correct properties to check for depending on the browser
    if ('sheet' in link) {
        sheet = 'sheet';
        cssRules = 'cssRules';
    } else {
        sheet = 'styleSheet';
        cssRules = 'rules';
    }

    var interval_id = setInterval(function () {                     // start checking whether the style sheet has successfully loaded
            try {
                if (link[sheet] && link[sheet][cssRules].length) { // SUCCESS! our style sheet has loaded
                    clearInterval(interval_id);                      // clear the counters
                    clearTimeout(timeout_id);
                    fn.call(scope || window, true, link);           // fire the callback with success == true
                }
            } catch (e) {
            } finally {
            }
        }, 50),                                                   // how often to check if the stylesheet is loaded
        timeout_id = setTimeout(function () {       // start counting down till fail
            clearInterval(interval_id);             // clear the counters
            clearTimeout(timeout_id);
            head.removeChild(link);                // since the style sheet didn't load, remove the link node from the DOM
            fn.call(scope || window, false, link); // fire the callback with success == false
        }, 15000);                                 // how long to wait before failing

    head.appendChild(link);  // insert the link node into the DOM and start loading the style sheet

    return link; // return the link node;
};
/* eslint-enable */

CatboostIpython.prototype.resizeCharts = function () {
    // width fix for development
    $('.catboost-graph__charts', this.layout).css({width: $('.catboost-graph').width()});

    this.plotly.Plots.resize(this.traces[this.activeTab].parent);
};

CatboostIpython.prototype.addMeta = function (path, meta) {
    this.meta[path] = meta;
};

CatboostIpython.prototype.addLayout = function (parent) {
    if (this.layout) {
        return;
    }

    var cvAreaControls = '';

    if (this.hasCVMode) {
        cvAreaControls = '<div>' +
            '<input type="checkbox" class="catboost-panel__control_checkbox" id="catboost-control2-cvstddev' + this.index + '"' + (this.stddevEnabled ? ' checked="checked"' : '') + '></input>' +
            '<label for="catboost-control2-cvstddev' + this.index + '" class="catboost-panel__controls2_label catboost-panel__controls2_label-long">Standard Deviation</label>' +
            '</div>';
    }

    this.layout = $('<div class="catboost">' +
        '<div class="catboost-panel">' +
        '<div class="catboost-panel__controls">' +
        '<input type="checkbox" class="catboost-panel__controls_checkbox" id="catboost-control-learn' + this.index + '" ' + (!this.layoutDisabled.learn ? ' checked="checked"' : '') + '></input>' +
        '<label for="catboost-control-learn' + this.index + '" class="catboost-panel__controls_label"><div class="catboost-panel__serie_learn_pic" style="border-color:#999"></div>Learn</label>' +
        '<input type="checkbox" class="catboost-panel__controls_checkbox" id="catboost-control-test' + this.index + '" ' + (!this.layoutDisabled.test ? ' checked="checked"' : '') + '></input>' +
        '<label for="catboost-control-test' + this.index + '" class="catboost-panel__controls_label"><div class="catboost-panel__serie_test_pic" style="border-color:#999"></div>Eval</label>' +
        '</div>' +
        '<div class="catboost-panel__series ' + (this.layoutDisabled.learn ? ' catboost-panel__series_learn_disabled' : '') + '">' +
        '</div>' +
        '<div class="catboost-panel__controls2">' +
        '<input type="checkbox" class="catboost-panel__control_checkbox" id="catboost-control2-clickmode' + this.index + '"></input>' +
        '<label for="catboost-control2-clickmode' + this.index + '" class="catboost-panel__controls2_label">Click Mode</label>' +
        '<input type="checkbox" class="catboost-panel__control_checkbox" id="catboost-control2-log' + this.index + '"></input>' +
        '<label for="catboost-control2-log' + this.index + '" class="catboost-panel__controls2_label">Logarithm</label>' +
        '<div>' +
        '<input type="checkbox" class="catboost-panel__control_checkbox" id="catboost-control2-smooth' + this.index + '"></input>' +
        '<label for="catboost-control2-smooth' + this.index + '" class="catboost-panel__controls2_label">Smooth</label>' +
        '<input id="catboost-control2-slider' + this.index + '" disabled="disabled" class="catboost-panel__control_slider" type ="range" value="0" min="0" max="1" step ="0.01" for="rangeInputValue" name="rangeInput"/>' +
        '<input id="catboost-control2-slidervalue' + this.index + '" disabled="disabled" class="catboost-panel__control_slidervalue" value="0" min="0" max="1" for="rangeInput" name="rangeInputValue"/>' +
        '</div>' +
        cvAreaControls +
        '</div>' +
        '</div>' +
        '<div class="catboost-graph">' +
        '<div class="catboost-graph__tabs"></div>' +
        '<div class="catboost-graph__charts"></div>' +
        '</div>' +
        '</div>');
    $(parent).append(this.layout);

    this.addTabEvents();
    this.addControlEvents();
};

CatboostIpython.prototype.addTabEvents = function () {
    var self = this;

    $('.catboost-graph__tabs', this.layout).click(function (e) {
        if (!$(e.target).is('.catboost-graph__tab:not(.catboost-graph__tab_active)')) {
            return;
        }

        var id = $(e.target).attr('tabid');

        self.activeTab = id;

        $('.catboost-graph__tab_active', self.layout).removeClass('catboost-graph__tab_active');
        $('.catboost-graph__chart_active', self.layout).removeClass('catboost-graph__chart_active');

        $('.catboost-graph__tab[tabid="' + id + '"]', self.layout).addClass('catboost-graph__tab_active');
        $('.catboost-graph__chart[tabid="' + id + '"]', self.layout).addClass('catboost-graph__chart_active');

        self.cleanSeries();

        self.redrawActiveChart();
        self.resizeCharts();
    });
};

CatboostIpython.prototype.addControlEvents = function () {
    var self = this;

    $('#catboost-control-learn' + this.index, this.layout).click(function () {
        self.layoutDisabled.learn = !$(this)[0].checked;

        $('.catboost-panel__series', self.layout).toggleClass('catboost-panel__series_learn_disabled', self.layoutDisabled.learn);

        self.redrawActiveChart();
    });

    $('#catboost-control-test' + this.index, this.layout).click(function () {
        self.layoutDisabled.test = !$(this)[0].checked;

        $('.catboost-panel__series', self.layout).toggleClass('catboost-panel__series_test_disabled', self.layoutDisabled.test);

        self.redrawActiveChart();
    });

    $('#catboost-control2-clickmode' + this.index, this.layout).click(function () {
        self.clickMode = $(this)[0].checked;
    });

    $('#catboost-control2-log' + this.index, this.layout).click(function () {
        self.logarithmMode = $(this)[0].checked ? 'log' : 'linear';

        self.forEveryLayout(function (layout) {
            layout.yaxis = {type: self.logarithmMode};
        });

        self.redrawActiveChart();
    });

    var slider = $('#catboost-control2-slider' + this.index),
        sliderValue = $('#catboost-control2-slidervalue' + this.index);

    $('#catboost-control2-smooth' + this.index, this.layout).click(function () {
        var enabled = $(this)[0].checked;

        self.setSmoothness(enabled ? self.lastSmooth : -1);

        slider.prop('disabled', !enabled);
        sliderValue.prop('disabled', !enabled);

        self.redrawActiveChart();
    });

    $('#catboost-control2-cvstddev' + this.index, this.layout).click(function () {
        var enabled = $(this)[0].checked;

        self.setStddev(enabled);

        self.redrawActiveChart();
    });

    slider.on('input change', function () {
        var smooth = Number($(this).val());

        sliderValue.val(isNaN(smooth) ? 0 : smooth);

        self.setSmoothness(smooth);
        self.lastSmooth = smooth;

        self.redrawActiveChart();
    });

    sliderValue.on('input change', function () {
        var smooth = Number($(this).val());

        slider.val(isNaN(smooth) ? 0 : smooth);

        self.setSmoothness(smooth);
        self.lastSmooth = smooth;

        self.redrawActiveChart();
    });
};

CatboostIpython.prototype.setTraceVisibility = function (trace, visibility) {
    if (trace) {
        trace.visible = visibility;
    }
};

CatboostIpython.prototype.updateTracesVisibility = function () {
    var tracesHash = this.groupTraces(),
        traces,
        smoothDisabled = this.getSmoothness() === -1,
        self = this;

    for (var train in tracesHash) {
        if (tracesHash.hasOwnProperty(train)) {
            traces = tracesHash[train].traces;

            if (this.layoutDisabled.traces[train]) {
                traces.forEach(function (trace) {
                    self.setTraceVisibility(trace, false);
                });
            } else {
                traces.forEach(function (trace) {
                    self.setTraceVisibility(trace, true);
                });

                if (this.hasCVMode) {
                    if (this.stddevEnabled) {
                        self.filterTracesOne(traces, {type: 'learn'}).forEach(function (trace) {
                            self.setTraceVisibility(trace, false);
                        });
                        self.filterTracesOne(traces, {type: 'test'}).forEach(function (trace) {
                            self.setTraceVisibility(trace, false);
                        });

                        self.filterTracesEvery(traces, this.getTraceDefParams({
                            type: 'learn',
                            cv_avg: true
                        })).forEach(function (trace) {
                            self.setTraceVisibility(trace, true);
                        });
                        self.filterTracesEvery(traces, this.getTraceDefParams({
                            type: 'test',
                            cv_avg: true
                        })).forEach(function (trace) {
                            self.setTraceVisibility(trace, true);
                        });

                        self.filterTracesEvery(traces, this.getTraceDefParams({
                            type: 'learn',
                            cv_avg: true,
                            smoothed: true
                        })).forEach(function (trace) {
                            self.setTraceVisibility(trace, true);
                        });
                        self.filterTracesEvery(traces, this.getTraceDefParams({
                            type: 'test',
                            cv_avg: true,
                            smoothed: true
                        })).forEach(function (trace) {
                            self.setTraceVisibility(trace, true);
                        });

                        self.filterTracesEvery(traces, this.getTraceDefParams({
                            type: 'test',
                            cv_avg: true,
                            best_point: true
                        })).forEach(function (trace) {
                            self.setTraceVisibility(trace, true);
                        });

                        self.filterTracesOne(traces, {cv_stddev_first: true}).forEach(function (trace) {
                            self.setTraceVisibility(trace, true);
                        });
                        self.filterTracesOne(traces, {cv_stddev_last: true}).forEach(function (trace) {
                            self.setTraceVisibility(trace, true);
                        });
                    } else {
                        self.filterTracesOne(traces, {cv_stddev_first: true}).forEach(function (trace) {
                            self.setTraceVisibility(trace, false);
                        });
                        self.filterTracesOne(traces, {cv_stddev_last: true}).forEach(function (trace) {
                            self.setTraceVisibility(trace, false);
                        });

                        self.filterTracesEvery(traces, this.getTraceDefParams({
                            type: 'learn',
                            cv_avg: true
                        })).forEach(function (trace) {
                            self.setTraceVisibility(trace, false);
                        });
                        self.filterTracesEvery(traces, this.getTraceDefParams({
                            type: 'test',
                            cv_avg: true
                        })).forEach(function (trace) {
                            self.setTraceVisibility(trace, false);
                        });

                        self.filterTracesEvery(traces, this.getTraceDefParams({
                            type: 'learn',
                            cv_avg: true,
                            smoothed: true
                        })).forEach(function (trace) {
                            self.setTraceVisibility(trace, false);
                        });
                        self.filterTracesEvery(traces, this.getTraceDefParams({
                            type: 'test',
                            cv_avg: true,
                            smoothed: true
                        })).forEach(function (trace) {
                            self.setTraceVisibility(trace, false);
                        });

                        self.filterTracesEvery(traces, this.getTraceDefParams({
                            type: 'test',
                            cv_avg: true,
                            best_point: true
                        })).forEach(function (trace) {
                            self.setTraceVisibility(trace, false);
                        });
                    }
                }

                if (smoothDisabled) {
                    self.filterTracesOne(traces, {smoothed: true}).forEach(function (trace) {
                        self.setTraceVisibility(trace, false);
                    });
                }

                if (this.layoutDisabled['learn']) {
                    self.filterTracesOne(traces, {type: 'learn'}).forEach(function (trace) {
                        self.setTraceVisibility(trace, false);
                    });
                }

                if (this.layoutDisabled['test']) {
                    self.filterTracesOne(traces, {type: 'test'}).forEach(function (trace) {
                        self.setTraceVisibility(trace, false);
                    });
                }
            }
        }
    }
};

CatboostIpython.prototype.getSmoothness = function () {
    return this.smoothness && this.smoothness > -1 ? this.smoothness : -1;
};

CatboostIpython.prototype.setSmoothness = function (weight) {
    if (weight < 0 && weight !== -1 || weight > 1) {
        return;
    }

    this.smoothness = weight;
};

CatboostIpython.prototype.setStddev = function (enabled) {
    this.stddevEnabled = enabled;
};

CatboostIpython.prototype.redrawActiveChart = function () {
    this.chartsToRedraw[this.activeTab] = true;

    this.redrawAll();
};

CatboostIpython.prototype.redraw = function () {
    if (this.chartsToRedraw[this.activeTab]) {
        this.chartsToRedraw[this.activeTab] = false;

        this.updateTracesVisibility();
        this.updateTracesCV();
        this.updateTracesBest();
        this.updateTracesValues();
        this.updateTracesSmoothness();

        this.plotly.redraw(this.traces[this.activeTab].parent);
    }

    this.drawTraces();
};

CatboostIpython.prototype.addRedrawFunc = function () {
    this.redrawFunc = throttle(this.redraw, 400, false, this);
};

CatboostIpython.prototype.redrawAll = function () {
    if (!this.redrawFunc) {
        this.addRedrawFunc();
    }

    this.redrawFunc();
};

CatboostIpython.prototype.addPoints = function (parent, data) {
    var self = this;

    data.chunks.forEach(function (item) {
        if (typeof item.remaining_time !== 'undefined' && typeof item.passed_time !== 'undefined') {
            if (!self.timeLeft[data.path]) {
                self.timeLeft[data.path] = [];
            }

            self.timeLeft[data.path][item.iteration] = [item.remaining_time, item.passed_time];
        }

        ['test', 'learn'].forEach(function (type) {
            var sets = self.meta[data.path][type + '_sets'],
                metrics = self.meta[data.path][type + '_metrics'];

            for (var i = 0; i < metrics.length; i++) {
                var nameOfMetric = metrics[i].name,
                    cvAdded = false;
                hovertextParametersAdded = false;

                self.lossFuncs[nameOfMetric] = metrics[i].best_value;

                for (var j = 0; j < sets.length; j++) {
                    var nameOfSet = sets[j],
                        params = {
                            chartName: nameOfMetric,
                            index: i,
                            train: data.train,
                            type: type,
                            path: data.path,
                            indexOfSet: j,
                            nameOfSet: nameOfSet
                        },
                        key = self.getKey(params),
                        launchMode = self.getLaunchMode(data.path);

                    if (!self.activeTab) {
                        self.activeTab = key.chartId;
                    }

                    if (launchMode === 'CV') {
                        // we need to set launch mode before first getTrace call
                        self.hasCVMode = true;

                        if (!self.isCVinited) {
                            // and we don't need to reset setting for next iterations
                            self.layoutDisabled.learn = true;
                            self.setStddev(true);

                            self.isCVinited = true;
                        }
                    }

                    var valuesOfSet = item[nameOfSet],
                        pointValue = valuesOfSet[i],
                        pointIndex = item.iteration,
                        // traces
                        trace = self.getTrace(parent, params),
                        smoothedTrace = self.getTrace(parent, $.extend({smoothed: true}, params)),
                        bestValueTrace = null;

                    if (type === 'test') {
                        if (launchMode !== 'CV') {
                            self.getTrace(parent, $.extend({best_point: true}, params));
                        }

                        if (typeof self.lossFuncs[nameOfMetric] === 'number') {
                            bestValueTrace = self.getTrace(parent, $.extend({best_value: true}, params));
                        }
                    }

                    if (pointValue !== 'inf' && pointValue !== 'nan') {
                        trace.x[pointIndex] = pointIndex;
                        trace.y[pointIndex] = valuesOfSet[i];
                        trace.hovertext[pointIndex] = nameOfSet + ': ' + valuesOfSet[i].toPrecision(7);
                        if (item.hasOwnProperty('parameters')) {
                            self.hovertextParameters[pointIndex] = '';
                            for (var parameter in item.parameters[0]) {
                                if (item.parameters[0].hasOwnProperty(parameter)) {
                                    valueOfParameter = item.parameters[0][parameter];
                                    self.hovertextParameters[pointIndex] += '<br>' + parameter + ' : ' + valueOfParameter;
                                }
                            }
                            if (!hovertextParametersAdded && type === 'test') {
                                hovertextParametersAdded = true;
                                trace.hovertext[pointIndex] += self.hovertextParameters[pointIndex];
                            }
                        }
                        smoothedTrace.x[pointIndex] = pointIndex;
                    }

                    if (bestValueTrace) {
                        bestValueTrace.x[pointIndex] = pointIndex;
                        bestValueTrace.y[pointIndex] = self.lossFuncs[nameOfMetric];
                    }

                    if (launchMode === 'CV' && !cvAdded) {
                        cvAdded = true;

                        self.getTrace(parent, $.extend({cv_stddev_first: true}, params));
                        self.getTrace(parent, $.extend({cv_stddev_last: true}, params));

                        self.getTrace(parent, $.extend({cv_stddev_first: true, smoothed: true}, params));
                        self.getTrace(parent, $.extend({cv_stddev_last: true, smoothed: true}, params));

                        self.getTrace(parent, $.extend({cv_avg: true}, params));
                        self.getTrace(parent, $.extend({cv_avg: true, smoothed: true}, params));

                        if (type === 'test') {
                            self.getTrace(parent, $.extend({cv_avg: true, best_point: true}, params));
                        }
                    }
                }

                self.chartsToRedraw[key.chartId] = true;

                self.redrawAll();
            }
        });
    });
};

CatboostIpython.prototype.getLaunchMode = function (path) {
    return this.meta[path].launch_mode;
};

CatboostIpython.prototype.getChartNode = function (params, active) {
    var node = $('<div class="catboost-graph__chart" tabid="' + params.id + '"></div>');

    if (active) {
        node.addClass('catboost-graph__chart_active');
    }

    return node;
};

CatboostIpython.prototype.getChartTab = function (params, active) {
    var node = $('<div class="catboost-graph__tab" tabid="' + params.id + '">' + params.name + '</div>');

    if (active) {
        node.addClass('catboost-graph__tab_active');
    }

    return node;
};

CatboostIpython.prototype.forEveryChart = function (callback) {
    for (var name in this.traces) {
        if (this.traces.hasOwnProperty(name)) {
            callback(this.traces[name]);
        }
    }
};

CatboostIpython.prototype.forEveryLayout = function (callback) {
    this.forEveryChart(function (chart) {
        callback(chart.layout);
    });
};

CatboostIpython.prototype.getChart = function (parent, params) {
    var id = params.id,
        self = this;

    if (this.charts[id]) {
        return this.charts[id];
    }

    this.addLayout(parent);

    var active = this.activeTab === params.id,
        chartNode = this.getChartNode(params, active),
        chartTab = this.getChartTab(params, active);

    $('.catboost-graph__charts', this.layout).append(chartNode);
    $('.catboost-graph__tabs', this.layout).append(chartTab);

    this.traces[id] = {
        id: params.id,
        name: params.name,
        parent: chartNode[0],
        traces: [],
        layout: {
            xaxis: {
                range: [0, Number(this.meta[params.path].iteration_count)],
                type: 'linear',
                tickmode: 'auto',
                showspikes: true,
                spikethickness: 1,
                spikedash: 'longdashdot',
                spikemode: 'across',
                zeroline: false,
                showgrid: false
            },
            yaxis: {
                zeroline: false
                //showgrid: false
                //hoverformat : '.7f'
            },
            separators: '. ',
            //hovermode: 'x',
            margin: {l: 38, r: 0, t: 35, b: 30},
            autosize: true,
            showlegend: false
        },
        options: {
            scrollZoom: false,
            modeBarButtonsToRemove: ['toggleSpikelines'],
            displaylogo: false
        }
    };

    this.charts[id] = this.plotly.plot(chartNode[0], this.traces[id].traces, this.traces[id].layout, this.traces[id].options);

    chartNode[0].on('plotly_hover', function (e) {
        self.updateTracesValues(e.points[0].x);
    });

    chartNode[0].on('plotly_click', function (e) {
        self.updateTracesValues(e.points[0].x, true);
    });

    return this.charts[id];
};


CatboostIpython.prototype.getTrace = function (parent, params) {
    var key = this.getKey(params),
        chartSeries = [];

    if (this.traces[key.chartId]) {
        chartSeries = this.traces[key.chartId].traces.filter(function (trace) {
            return trace.name === key.traceName;
        });
    }

    if (chartSeries.length) {
        return chartSeries[0];
    } else {
        this.getChart(parent, {id: key.chartId, name: params.chartName, path: params.path});

        var plotParams = {
                color: this.getNextColor(params.path, params.smoothed ? 0.2 : 1),
                fillsmoothcolor: this.getNextColor(params.path, 0.1),
                fillcolor: this.getNextColor(params.path, 0.4),
                hoverinfo: params.cv_avg ? 'skip' : 'text+x',
                width: params.cv_avg ? 2 : 1,
                dash: params.type === 'test' ? 'solid' : 'dot'
            },
            trace = {
                name: key.traceName,
                _params: params,
                x: [],
                y: [],
                hovertext: [],
                hoverinfo: plotParams.hoverinfo,
                line: {
                    width: plotParams.width,
                    dash: plotParams.dash,
                    color: plotParams.color
                },
                mode: 'lines',
                hoveron: 'points',
                connectgaps: true
            };

        if (params.best_point) {
            trace = {
                name: key.traceName,
                _params: params,
                x: [],
                y: [],
                marker: {
                    width: 2,
                    color: plotParams.color
                },
                hovertext: [],
                hoverinfo: 'text',
                mode: 'markers',
                type: 'scatter'
            };
        }

        if (params.best_value) {
            trace = {
                name: key.traceName,
                _params: params,
                x: [],
                y: [],
                line: {
                    width: 1,
                    dash: 'dash',
                    color: '#CCCCCC'
                },
                mode: 'lines',
                connectgaps: true,
                hoverinfo: 'skip'
            };
        }

        if (params.cv_stddev_last) {
            trace.fill = 'tonexty';
        }

        trace._params.plotParams = plotParams;

        this.traces[key.chartId].traces.push(trace);

        return trace;
    }
};

CatboostIpython.prototype.getKey = function (params) {
    var traceName = [
        params.train,
        params.type,
        params.indexOfSet,
        (params.smoothed ? 'smoothed' : ''),
        (params.best_point ? 'best_pount' : ''),
        (params.best_value ? 'best_value' : ''),
        (params.cv_avg ? 'cv_avg' : ''),
        (params.cv_stddev_first ? 'cv_stddev_first' : ''),
        (params.cv_stddev_last ? 'cv_stddev_last' : '')
    ].join(';');

    return {
        chartId: params.chartName,
        traceName: traceName,
        colorId: params.train
    };
};

CatboostIpython.prototype.filterTracesEvery = function (traces, filter) {
    traces = traces || this.traces[this.activeTab].traces;

    return traces.filter(function (trace) {
        for (var prop in filter) {
            if (filter.hasOwnProperty(prop)) {
                if (filter[prop] !== trace._params[prop]) {
                    return false;
                }
            }
        }

        return true;
    });
};

CatboostIpython.prototype.filterTracesOne = function (traces, filter) {
    traces = traces || this.traces[this.activeTab].traces;

    return traces.filter(function (trace) {
        for (var prop in filter) {
            if (filter.hasOwnProperty(prop)) {
                if (filter[prop] === trace._params[prop]) {
                    return true;
                }
            }
        }

        return false;
    });
};

CatboostIpython.prototype.cleanSeries = function () {
    $('.catboost-panel__series', this.layout).html('');
};

CatboostIpython.prototype.groupTraces = function () {
    var traces = this.traces[this.activeTab].traces,
        index = 0,
        tracesHash = {};

    traces.map(function (trace) {
        var train = trace._params.train;

        if (!tracesHash[train]) {
            tracesHash[train] = {
                index: index,
                traces: [],
                info: {
                    path: trace._params.path,
                    color: trace._params.plotParams.color
                }
            };

            index++;
        }

        tracesHash[train].traces.push(trace);
    });

    return tracesHash;
};

CatboostIpython.prototype.drawTraces = function () {
    var html = '',
        tracesHash = this.groupTraces();

    var curLength = $('.catboost-panel__series .catboost-panel__serie', this.layout).length;
    var newLength = Object.keys(tracesHash).filter(hasOwnProperty.bind(tracesHash)).length;
    if (newLength === curLength) {
        return;
    }

    for (var train in tracesHash) {
        if (tracesHash.hasOwnProperty(train)) {
            html += this.drawTrace(train, tracesHash[train]);
        }
    }

    $('.catboost-panel__series', this.layout).html(html);

    this.updateTracesValues();

    this.addTracesEvents();
};

CatboostIpython.prototype.getTraceDefParams = function (params) {
    var defParams = {
        smoothed: undefined,
        best_point: undefined,
        best_value: undefined,
        cv_avg: undefined,
        cv_stddev_first: undefined,
        cv_stddev_last: undefined
    };

    if (params) {
        return $.extend(defParams, params);
    } else {
        return defParams;
    }
};

CatboostIpython.prototype.drawTrace = function (train, hash) {
    var info = hash.info,
        id = 'catboost-serie-' + this.index + '-' + hash.index,
        traces = {
            learn: this.filterTracesEvery(hash.traces, this.getTraceDefParams({type: 'learn'})),
            test: this.filterTracesEvery(hash.traces, this.getTraceDefParams({type: 'test'}))
        },
        items = {
            learn: {
                middle: '',
                bottom: ''
            },
            test: {
                middle: '',
                bottom: ''
            }
        },
        tracesNames = '';

    ['learn', 'test'].forEach(function (type) {
        traces[type].forEach(function (trace) {
            items[type].middle += '<div class="catboost-panel__serie_' + type + '_pic" style="border-color:' + info.color + '"></div>' +
                '<div data-index="' + trace._params.indexOfSet + '" class="catboost-panel__serie_' + type + '_value"></div>';

            items[type].bottom += '<div class="catboost-panel__serie_' + type + '_pic" style="border-color:transparent"></div>' +
                '<div data-index="' + trace._params.indexOfSet + '" class="catboost-panel__serie_best_' + type + '_value"></div>';

            tracesNames += '<div class="catboost-panel__serie_' + type + '_pic" style="border-color:' + info.color + '"></div>' +
                '<div class="catboost-panel__serie_' + type + '_name">' + trace._params.nameOfSet + '</div>';
        });
    });

    var timeSpendHtml = '<div class="catboost-panel__serie_time">' +
        '<div class="catboost-panel__serie_time_spend" title="Time spend"></div>' +
        '</div>';

    var html = '<div id="' + id + '" class="catboost-panel__serie" style="color:' + info.color + '">' +
        '<div class="catboost-panel__serie_top">' +
        '<input type="checkbox" data-seriename="' + train + '" class="catboost-panel__serie_checkbox" id="' + id + '-box" ' + (!this.layoutDisabled.series[train] ? 'checked="checked"' : '') + '></input>' +
        '<label title=' + this.meta[info.path].name + ' for="' + id + '-box" class="catboost-panel__serie_label">' + train + '<div class="catboost-panel__serie_time_left" title="Estimate time"></div></label>' +
        (this.getLaunchMode(info.path) !== 'Eval' ? timeSpendHtml : '') +
        '</div>' +
        '<div class="catboost-panel__serie_hint catboost-panel__serie__learn_hint">curr</div>' +
        '<div class="catboost-panel__serie_hint catboost-panel__serie__test_hint">best</div>' +
        '<div class="catboost-panel__serie_iteration" title="curr iteration"></div>' +
        '<div class="catboost-panel__serie_best_iteration" title="best ' + (this.hasCVMode ? 'avg ' : '') + 'iteration"></div>' +
        '<div class="catboost-panel__serie_scroll">' +
        '<div class="catboost-panel__serie_names">' +
        tracesNames +
        '</div>' +
        '<div class="catboost-panel__serie_middle">' +
        items.learn.middle +
        items.test.middle +
        '</div>' +
        '<div class="catboost-panel__serie_bottom">' +
        items.learn.bottom +
        items.test.bottom +
        '</div>' +
        '</div>' +
        '</div>';

    return html;
};

CatboostIpython.prototype.updateTracesValues = function (iteration, click) {
    var tracesHash = this.groupTraces();

    for (var train in tracesHash) {
        if (tracesHash.hasOwnProperty(train) && !this.layoutDisabled.traces[train]) {
            this.updateTraceValues(train, tracesHash[train], iteration, click);
        }
    }
};

CatboostIpython.prototype.updateTracesBest = function () {
    var tracesHash = this.groupTraces();

    for (var train in tracesHash) {
        if (tracesHash.hasOwnProperty(train) && !this.layoutDisabled.traces[train]) {
            this.updateTraceBest(train, tracesHash[train]);
        }
    }
};

CatboostIpython.prototype.getBestValue = function (data) {
    if (!data.length) {
        return {
            best: undefined,
            index: -1
        };
    }

    var best = data[0],
        index = 0,
        func = this.lossFuncs[this.traces[this.activeTab].name],
        bestDiff = typeof func === 'number' ? Math.abs(data[0] - func) : 0;

    for (var i = 1, l = data.length; i < l; i++) {
        if (func === 'Min' && data[i] < best) {
            best = data[i];
            index = i;
        }

        if (func === 'Max' && data[i] > best) {
            best = data[i];
            index = i;
        }

        if (typeof func === 'number' && Math.abs(data[i] - func) < bestDiff) {
            best = data[i];
            bestDiff = Math.abs(data[i] - func);
            index = i;
        }
    }

    return {
        best: best,
        index: index,
        func: func
    };
};

CatboostIpython.prototype.updateTracesCV = function () {
    this.updateTracesCVAvg();

    if (this.hasCVMode && this.stddevEnabled) {
        this.updateTracesCVStdDev();
    }
};

CatboostIpython.prototype.updateTracesCVAvg = function () {
    var tracesHash = this.groupTraces(),
        avgTraces = this.filterTracesEvery(tracesHash.traces, this.getTraceDefParams({
            cv_avg: true
        })),
        self = this;

    avgTraces.forEach(function (trace) {
        var origTraces = self.filterTracesEvery(tracesHash.traces, self.getTraceDefParams({
            train: trace._params.train,
            type: trace._params.type,
            smoothed: trace._params.smoothed
        }));

        if (origTraces.length) {
            self.cvAvgFunc(origTraces, trace);
        }
    });
};

CatboostIpython.prototype.cvAvgFunc = function (origTraces, avgTrace) {
    var maxCount = origTraces.length,
        maxLength = -1,
        count,
        sum;

    origTraces.forEach(function (origTrace) {
        if (origTrace.y.length > maxLength) {
            maxLength = origTrace.y.length;
        }
    });

    for (var i = 0; i < maxLength; i++) {
        sum = 0;
        count = 0;

        for (var j = 0; j < maxCount; j++) {
            if (typeof origTraces[j].y[i] !== 'undefined') {
                sum += origTraces[j].y[i];
                count++;
            }
        }

        if (count > 0) {
            avgTrace.x[i] = i;
            avgTrace.y[i] = sum / count;
        }
    }
};

CatboostIpython.prototype.updateTracesCVStdDev = function () {
    var tracesHash = this.groupTraces(),
        firstTraces = this.filterTracesOne(tracesHash.traces, {cv_stddev_first: true}),
        self = this;

    firstTraces.forEach(function (trace) {
        var origTraces = self.filterTracesEvery(tracesHash.traces, self.getTraceDefParams({
                train: trace._params.train,
                type: trace._params.type,
                smoothed: trace._params.smoothed
            })),
            lastTraces = self.filterTracesEvery(tracesHash.traces, self.getTraceDefParams({
                train: trace._params.train,
                type: trace._params.type,
                smoothed: trace._params.smoothed,
                cv_stddev_last: true
            }));

        if (origTraces.length && lastTraces.length === 1) {
            self.cvStdDevFunc(origTraces, trace, lastTraces[0]);
        }
    });
};

CatboostIpython.prototype.cvStdDevFunc = function (origTraces, firstTrace, lastTrace) {
    var maxCount = origTraces.length,
        maxLength = -1,
        count,
        sum,
        i, j;

    origTraces.forEach(function (origTrace) {
        if (origTrace.y.length > maxLength) {
            maxLength = origTrace.y.length;
        }
    });

    for (i = 0; i < maxLength; i++) {
        sum = 0;
        count = 0;

        for (j = 0; j < maxCount; j++) {
            if (typeof origTraces[j].y[i] !== 'undefined') {
                sum += origTraces[j].y[i];
                count++;
            }
        }

        if (count <= 0) {
            continue;
        }

        var std = 0,
            avg = sum / count;

        for (j = 0; j < maxCount; j++) {
            if (typeof origTraces[j].y[i] !== 'undefined') {
                std += Math.pow(origTraces[j].y[i] - avg, 2);
            }
        }

        std /= (count - 1);
        std = Math.pow(std, 0.5);

        firstTrace.x[i] = i;
        firstTrace.y[i] = avg - std;
        firstTrace.hovertext[i] = firstTrace._params.type + ' std: ' + avg.toFixed(7) + '-' + std.toFixed(7);
        lastTrace.x[i] = i;
        lastTrace.y[i] = avg + std;
        lastTrace.hovertext[i] = lastTrace._params.type + ' std: ' + avg.toFixed(7) + '+' + std.toFixed(7);
        if (this.hovertextParameters.length > i) {
            firstTrace.hovertext[i] += this.hovertextParameters[i];
            lastTrace.hovertext[i] += this.hovertextParameters[i];
        }
    }
};

CatboostIpython.prototype.updateTracesSmoothness = function () {
    var tracesHash = this.groupTraces(),
        smoothedTraces = this.filterTracesOne(tracesHash.traces, {smoothed: true}),
        enabled = this.getSmoothness() > -1,
        self = this;

    smoothedTraces.forEach(function (trace) {
        var origTraces = self.filterTracesEvery(tracesHash.traces, self.getTraceDefParams({
                train: trace._params.train,
                type: trace._params.type,
                indexOfSet: trace._params.indexOfSet,
                cv_avg: trace._params.cv_avg,
                cv_stddev_first: trace._params.cv_stddev_first,
                cv_stddev_last: trace._params.cv_stddev_last
            })),
            colorFlag = false;

        if (origTraces.length === 1) {
            origTraces = origTraces[0];

            if (origTraces.visible) {
                if (enabled) {
                    self.smoothFunc(origTraces, trace);
                    colorFlag = true;
                }

                self.highlightSmoothedTrace(origTraces, trace, colorFlag);
            }
        }
    });
};

CatboostIpython.prototype.highlightSmoothedTrace = function (trace, smoothedTrace, flag) {
    if (flag) {
        smoothedTrace.line.color = trace._params.plotParams.color;
        trace.line.color = smoothedTrace._params.plotParams.color;
        trace.hoverinfo = 'skip';

        if (trace._params.cv_stddev_last) {
            trace.fillcolor = trace._params.plotParams.fillsmoothcolor;
        }
    } else {
        trace.line.color = trace._params.plotParams.color;
        trace.hoverinfo = trace._params.plotParams.hoverinfo;

        if (trace._params.cv_stddev_last) {
            trace.fillcolor = trace._params.plotParams.fillcolor;
        }
    }
};

CatboostIpython.prototype.smoothFunc = function (origTrace, smoothedTrace) {
    var data = origTrace.y,
        smoothedPoints = this.smooth(data, this.getSmoothness()),
        smoothedIndex = 0,
        self = this;

    if (smoothedPoints.length) {
        data.forEach(function (d, index) {
            if (!smoothedTrace.x[index]) {
                smoothedTrace.x[index] = index;
            }

            var nameOfSet = smoothedTrace._params.nameOfSet;

            if (smoothedTrace._params.cv_stddev_first || smoothedTrace._params.cv_stddev_last) {
                nameOfSet = smoothedTrace._params.type + ' std';
            }

            smoothedTrace.y[index] = smoothedPoints[smoothedIndex];
            smoothedTrace.hovertext[index] = nameOfSet + '`: ' + smoothedPoints[smoothedIndex].toPrecision(7);
            if (self.hovertextParameters.length > index) {
                smoothedTrace.hovertext[index] += self.hovertextParameters[index];
            }
            smoothedIndex++;
        });
    }
};

CatboostIpython.prototype.formatItemValue = function (value, index, suffix) {
    if (typeof value === 'undefined') {
        return '';
    }

    suffix = suffix || '';

    return '<span title="' + suffix + 'value ' + value + '">' + value + '</span>';
};

CatboostIpython.prototype.updateTraceBest = function (train, hash) {
    var traces = this.filterTracesOne(hash.traces, {best_point: true}),
        self = this;

    traces.forEach(function (trace) {
        var testTrace = self.filterTracesEvery(hash.traces, self.getTraceDefParams({
            train: trace._params.train,
            type: 'test',
            indexOfSet: trace._params.indexOfSet
        }));

        if (self.hasCVMode) {
            testTrace = self.filterTracesEvery(hash.traces, self.getTraceDefParams({
                train: trace._params.train,
                type: 'test',
                cv_avg: true
            }));
        }

        var bestValue = self.getBestValue(testTrace.length === 1 ? testTrace[0].y : []);

        if (bestValue.index !== -1) {
            trace.x[0] = bestValue.index;
            trace.y[0] = bestValue.best;
            trace.hovertext[0] = bestValue.func + ' (' + (self.hasCVMode ? 'avg' : trace._params.nameOfSet) + '): ' + bestValue.index + ' ' + bestValue.best;
        }
    });
};

CatboostIpython.prototype.updateTraceValues = function (name, hash, iteration, click) {
    var id = 'catboost-serie-' + this.index + '-' + hash.index,
        traces = {
            learn: this.filterTracesEvery(hash.traces, this.getTraceDefParams({type: 'learn'})),
            test: this.filterTracesEvery(hash.traces, this.getTraceDefParams({type: 'test'}))
        },
        path = hash.info.path,
        self = this;

    ['learn', 'test'].forEach(function (type) {
        traces[type].forEach(function (trace) {
            var data = trace.y || [],
                index = typeof iteration !== 'undefined' && iteration < data.length - 1 ? iteration : data.length - 1,
                value = data.length ? data[index] : undefined,
                testTrace = self.filterTracesEvery(hash.traces, self.getTraceDefParams({
                    type: 'test',
                    indexOfSet: trace._params.indexOfSet
                })),
                bestValue = self.getBestValue(testTrace.length === 1 ? testTrace[0].y : []),
                timeLeft = '',
                timeSpend = '';

            if (click || !self.clickMode) {
                $('#' + id + ' .catboost-panel__serie_' + type + '_value[data-index=' + trace._params.indexOfSet + ']', self.layout)
                    .html(self.formatItemValue(value, index, type + ' '));
                $('#' + id + ' .catboost-panel__serie_iteration', self.layout).html(index);

                if (self.timeLeft[path] && self.timeLeft[path][data.length - 1]) {
                    timeLeft = self.timeLeft[path][data.length - 1][0];
                }
                $('#' + id + ' .catboost-panel__serie_time_left', self.layout).html(timeLeft ? ('~' + self.convertTime(timeLeft)) : '');

                if (self.timeLeft[path] && self.timeLeft[path][index]) {
                    timeSpend = self.timeLeft[path][index][1];
                }

                $('#' + id + ' .catboost-panel__serie_time_spend', self.layout).html(self.convertTime(timeSpend));
                $('#' + id + ' .catboost-panel__serie_best_iteration', self.layout).html(bestValue.index > -1 ? bestValue.index : '');

                $('#' + id + ' .catboost-panel__serie_best_test_value[data-index=' + trace._params.indexOfSet + ']', self.layout)
                    .html(self.formatItemValue(bestValue.best, bestValue.index, 'best ' + trace._params.nameOfSet + ' '));
            }
        });
    });

    if (this.hasCVMode) {
        var testTrace = this.filterTracesEvery(hash.traces, this.getTraceDefParams({
                type: 'test',
                cv_avg: true
            })),
            bestValue = this.getBestValue(testTrace.length === 1 ? testTrace[0].y : []);

        $('#' + id + ' .catboost-panel__serie_best_iteration', this.layout).html(bestValue.index > -1 ? bestValue.index : '');
    }

    if (click) {
        this.clickMode = true;

        $('#catboost-control2-clickmode' + this.index, this.layout)[0].checked = true;
    }
};

CatboostIpython.prototype.addTracesEvents = function () {
    var self = this;

    $('.catboost-panel__serie_checkbox', this.layout).click(function () {
        var name = $(this).data('seriename');

        self.layoutDisabled.traces[name] = !$(this)[0].checked;

        self.redrawActiveChart();
    });
};

CatboostIpython.prototype.getNextColor = function (path, opacity) {
    var color;

    if (this.colorsByPath[path]) {
        color = this.colorsByPath[path];
    } else {
        color = this.colors[this.colorIndex];
        this.colorsByPath[path] = color;

        this.colorIndex++;

        if (this.colorIndex > this.colors.length - 1) {
            this.colorIndex = 0;
        }
    }

    return this.hexToRgba(color, opacity);
};

CatboostIpython.prototype.hexToRgba = function (value, opacity) {
    if (value.length < 6) {
        var pattern = /^#?([a-f\d])([a-f\d])([a-f\d])/i;
        value = value.replace(pattern, function (m, r, g, b) {
            return '#' + r + r + g + g + b + b;
        });
    }

    var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})/i.exec(value);
    var rgb = {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
    };

    return 'rgba(' + rgb.r + ',' + rgb.g + ',' + rgb.b + ',' + opacity + ')';
};

CatboostIpython.prototype.convertTime = function (time) {
    if (!time) {
        return '0s';
    }

    time = Math.floor(time * 1000);

    var millis = time % 1000;
    time = parseInt(time / 1000, 10);
    var seconds = time % 60;
    time = parseInt(time / 60, 10);
    var minutes = time % 60;
    time = parseInt(time / 60, 10);
    var hours = time % 24;
    var out = '';
    if (hours && hours > 0) {
        out += hours + 'h ';
        seconds = 0;
        millis = 0;
    }
    if (minutes && minutes > 0) {
        out += minutes + 'm ';
        millis = 0;
    }
    if (seconds && seconds > 0) {
        out += seconds + 's ';
    }
    if (millis && millis > 0) {
        out += millis + 'ms';
    }

    return out.trim();
};

CatboostIpython.prototype.mean = function (values, valueof) {
    var n = values.length,
        m = n,
        i = -1,
        value,
        sum = 0,
        number = function (x) {
            return x === null ? NaN : +x;
        };

    if (valueof === null) {
        while (++i < n) {
            if (!isNaN(value = number(values[i]))) {
                sum += value;
            } else {
                --m;
            }
        }
    } else {
        while (++i < n) {
            if (!isNaN(value = number(valueof(values[i], i, values)))) {
                sum += value;
            } else {
                --m;
            }
        }
    }

    if (m) {
        return sum / m;
    }
};

// from TensorBoard
CatboostIpython.prototype.smooth = function (data, weight) {
    // When increasing the smoothing window, it smoothes a lot with the first
    // few points and then starts to gradually smooth slower, so using an
    // exponential function makes the slider more consistent. 1000^x has a
    // range of [1, 1000], so subtracting 1 and dividing by 999 results in a
    // range of [0, 1], which can be used as the percentage of the data, so
    // that the kernel size can be specified as a percentage instead of a
    // hardcoded number, what would be bad with multiple series.
    var factor = (Math.pow(1000, weight) - 1) / 999,
        kernelRadius = Math.floor(data.length * factor / 2),
        res = [],
        self = this;

    data.forEach(function (d, i) {
        var actualKernelRadius = Math.min(kernelRadius, i, data.length - i - 1);
        var start = i - actualKernelRadius;
        var end = i + actualKernelRadius + 1;
        var point = d;
        // Only smooth finite numbers.
        if (!isFinite(point)) {
            res.push(point);
        } else {
            res.push(self.mean(data.slice(start, end).filter(function (d) {
                return isFinite(d);
            }), null));
        }
    });

    return res;
};

var getInstance = function (el) {
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
    addInstance = function (el) {
        $(el).attr('catboost-id', 'catboost_' + window.catboostIpythonIndex);

        var catboostIpython = new CatboostIpython();
        catboostIpython.index = catboostIpythonIndex;
        catboostIpython.plotly = Plotly;

        window.catboostIpythonInstances[window.catboostIpythonIndex] = catboostIpython;

        window.catboostIpythonIndex++;

        return catboostIpython;
    };

class CatboostIpythonWidget extends widgets.DOMWidgetView {
    initialize() {
        widgets.DOMWidgetView.prototype.initialize.apply(this, arguments);

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
    }

    render() {
        this.value_changed();
        this.model.on('change:value', this.value_changed, this);
    }

    update() {
        this.value_changed();
    }

    value_changed() {
        this.el.style['width'] = this.model.get('width');
        this.el.style['height'] = this.model.get('height');
        this.displayed.then(_.bind(this.render_charts, this));
    }

    process_all(parent, params) {
        var data = params.data;

        for (var path in data) {
            if (data.hasOwnProperty(path)) {
                this.process_row(parent, data[path]);
            }
        }
    }

    process_row(parent, data) {
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
            catboostIpython.lastIndex = {};
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
    }

    render_charts() {
        this.process_all(this.el, {
            data: this.model.get('data')
        });

        return this;
    }
}

class CatboostWidgetModel extends widgets.DOMWidgetModel {
    defaults() {
        return Object.assign(
            {},
            super.defaults(),
            {
                _model_name: 'CatboostWidgetModel',
                _view_name: 'CatboostWidgetView',
                _model_module: 'catboost-widget',
                _view_module: 'catboost-widget',
                _model_module_version: widget_version,
                _view_module_version: widget_version,
            }
        )
    }
}

module.exports = {
    CatboostWidgetModel: CatboostWidgetModel,
    CatboostWidgetView: CatboostIpythonWidget
};
