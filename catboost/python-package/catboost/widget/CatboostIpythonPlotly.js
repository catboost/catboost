/* global Highcharts, Plotly*/
/* eslint-disable */
function throttle(fn, timeout, invokeAsap, ctx) {
    var typeofInvokeAsap = typeof invokeAsap;
    if(typeofInvokeAsap === 'undefined') {
        invokeAsap = true;
    } else if(arguments.length === 3 && typeofInvokeAsap !== 'boolean') {
        ctx = invokeAsap;
        invokeAsap = true;
    }

    var timer, args, needInvoke,
        wrapper = function() {
            if(needInvoke) {
                fn.apply(ctx, args);
                needInvoke = false;
                timer = setTimeout(wrapper, timeout);
            } else {
                timer = null;
            }
        };

    return function() {
        args = arguments;
        ctx || (ctx = this);
        needInvoke = true;

        if(!timer) {
            invokeAsap?
                wrapper() :
                timer = setTimeout(wrapper, timeout);
        }
    };
};

/* eslint-enable */
function CatboostIpython() {

}

CatboostIpython.prototype.init = function() {
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
                        name: "1_learn",
                        x: [],
                        y: []
                    },
                    {
                        name: "1_learn__smoothed__",
                        x: [],
                        y: []
                    },
                    {
                        name: "1_test",
                        x: [],
                        y: []
                    },
                    {
                        name: "1_test__smoothed__",
                        x: [],
                        y: []
                    },
                    {
                        name: "1_test__min__",
                        x: [],
                        y: []
                    }
                ]
            }
        }
    */

    this.chartsToRedraw = {};
    this.lastIndexes = {};
    this.smoothness = -1;
    this.layoutDisabled = {
        series: {}
    };
    this.clickMode = false;
    this.logarithmMode = 'linear';
    this.lastSmooth = 0.5;
    this.layout = null;
    this.activeTab = '';
    this.meta = {};
    this.timeLeft = {};

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
};

/* eslint-disable */
CatboostIpython.prototype.loadStyles = function(path, fn, scope) {
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
        sheet = 'sheet'; cssRules = 'cssRules';
    } else {
        sheet = 'styleSheet'; cssRules = 'rules';
    }

    var interval_id = setInterval(function() {                     // start checking whether the style sheet has successfully loaded
        try {
            if (link[sheet] && link[sheet][cssRules].length) { // SUCCESS! our style sheet has loaded
                clearInterval(interval_id);                      // clear the counters
                clearTimeout(timeout_id);
                fn.call( scope || window, true, link);           // fire the callback with success == true
            }
        } catch(e) {} finally {}
    }, 50 ),                                                   // how often to check if the stylesheet is loaded
    timeout_id = setTimeout( function() {       // start counting down till fail
        clearInterval( interval_id );             // clear the counters
        clearTimeout( timeout_id );
        head.removeChild( link );                // since the style sheet didn't load, remove the link node from the DOM
        fn.call( scope || window, false, link ); // fire the callback with success == false
    }, 15000 );                                 // how long to wait before failing

    head.appendChild( link );  // insert the link node into the DOM and start loading the style sheet

    return link; // return the link node;
};
/* eslint-enable */

CatboostIpython.prototype.resizeCharts = function() {
    // width fix for development
    $('.catboost-graph__charts', this.layout).css({width: $('.catboost-graph').width()});

    this.plotly.Plots.resize(this.traces[this.activeTab].parent);
};

CatboostIpython.prototype.addMeta = function(path, meta) {
    this.meta[path] = meta;
};

CatboostIpython.prototype.setTime = function(path, timeLeft) {
    this.timeLeft[path] = timeLeft;
};

CatboostIpython.prototype.addLayout = function(parent) {
    if (this.layout) {
        return;
    }

    this.layout = $('<div class="catboost">' +
                        '<div class="catboost-panel">' +
                            '<div class="catboost-panel__controls">' +
                                '<input type="checkbox" class="catboost-panel__controls_checkbox" id="catboost-control-learn' + this.index + '" checked="checked"></input>' +
                                '<label for="catboost-control-learn' + this.index + '" class="catboost-panel__controls_label"><div class="catboost-panel__serie_learn_pic" style="border-color:#999"></div>Learn</label>' +
                                '<input type="checkbox" class="catboost-panel__controls_checkbox" id="catboost-control-test' + this.index + '" checked="checked"></input>' +
                                '<label for="catboost-control-test' + this.index + '" class="catboost-panel__controls_label"><div class="catboost-panel__serie_test_pic" style="border-color:#999"></div>Test</label>' +
                            '</div>' +
                            '<div class="catboost-panel__series">' +
                            '</div>' +
                            '<div class="catboost-panel__controls2">' +
                                '<input type="checkbox" class="catboost-panel__control_checkbox" id="catboost-control2-clickmode' + this.index + '"></input>' +
                                '<label for="catboost-control2-clickmode' + this.index + '" class="catboost-panel__controls2_label">Click Mode</label>' +
                                '<input type="checkbox" class="catboost-panel__control_checkbox" id="catboost-control2-log' + this.index + '"></input>' +
                                '<label for="catboost-control2-log' + this.index + '" class="catboost-panel__controls2_label">Logarithm</label>' +
                                '<div>' +
                                    '<input type="checkbox" class="catboost-panel__control_checkbox" id="catboost-control2-smooth' + this.index + '"></input>' +
                                    '<label for="catboost-control2-smooth' + this.index + '" class="catboost-panel__controls2_label">Smooth</label>' +
                                    '<input id="catboost-control2-slider' + this.index + '" disabled="disabled" class="catboost-panel__control_slider" type ="range" value="0.5" min="0" max="1" step ="0.01" for="rangeInputValue" name="rangeInput"/>' +
                                    '<input id="catboost-control2-slidervalue' + this.index + '" disabled="disabled" class="catboost-panel__control_slidervalue" value="0.5" min="0" max="1" for="rangeInput" name="rangeInputValue"/>' +
                                '</div>' +
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

CatboostIpython.prototype.addTabEvents = function() {
    var self = this;

    $('.catboost-graph__tabs', this.layout).click(function(e) {
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
    });
};

CatboostIpython.prototype.addControlEvents = function() {
    var self = this;

    $('#catboost-control-learn' + this.index, this.layout).click(function() {
        self.layoutDisabled.learn = !$(this)[0].checked;

        $('.catboost-panel__series', self.layout).toggleClass('catboost-panel__series_learn_disabled', self.layoutDisabled.learn);

        self.redrawActiveChart();
    });

    $('#catboost-control-test' + this.index, this.layout).click(function() {
        self.layoutDisabled.test = !$(this)[0].checked;

        $('.catboost-panel__series', self.layout).toggleClass('catboost-panel__series_test_disabled', self.layoutDisabled.test);

        self.redrawActiveChart();
    });

    $('#catboost-control2-clickmode' + this.index, this.layout).click(function() {
        self.clickMode = $(this)[0].checked;
    });

    $('#catboost-control2-log' + this.index, this.layout).click(function() {
        self.logarithmMode = $(this)[0].checked ? 'log' : 'linear';

        self.forEveryLayout(function(layout) {
            layout.yaxis = {type: self.logarithmMode};
        });

        self.redrawActiveChart();
    });

    var slider = $('#catboost-control2-slider' + this.index),
        sliderValue = $('#catboost-control2-slidervalue' + this.index);

    $('#catboost-control2-smooth' + this.index, this.layout).click(function() {
        var enabled = $(this)[0].checked;

        self.setSmoothness(enabled ? self.lastSmooth : -1);

        slider.prop('disabled', !enabled);
        sliderValue.prop('disabled', !enabled);

        self.redrawActiveChart();
    });

    slider.on('input change', function() {
        var smooth = Number($(this).val());

        sliderValue.val(isNaN(smooth) ? 0 : smooth);

        self.setSmoothness(smooth);
        self.lastSmooth = smooth;

        self.redrawActiveChart();
    });

    sliderValue.on('input change', function() {
        var smooth = Number($(this).val());

        slider.val(isNaN(smooth) ? 0 : smooth);

        self.setSmoothness(smooth);
        self.lastSmooth = smooth;

        self.redrawActiveChart();
    });
};

CatboostIpython.prototype.setSerieVisibility = function(serie, visibility) {
    if (serie) {
        serie.visible = visibility;
    }
};

CatboostIpython.prototype.updateSeriesVisibility = function() {
    var seriesHash = this.groupSeries(),
        series;

    for (var id in seriesHash) {
        if (seriesHash.hasOwnProperty(id)) {
            series = seriesHash[id].series;

            if (this.layoutDisabled.series[id]) {
                this.setSerieVisibility(series.learn, false);
                this.setSerieVisibility(series.learn__smoothed__, false);

                this.setSerieVisibility(series.test, false);
                this.setSerieVisibility(series.test__smoothed__, false);

                this.setSerieVisibility(series.test__min__, false);
            } else {
                this.setSerieVisibility(series.learn, true);
                this.setSerieVisibility(series.learn__smoothed__, true);

                this.setSerieVisibility(series.test, true);
                this.setSerieVisibility(series.test__smoothed__, true);

                this.setSerieVisibility(series.test__min__, true);

                if (this.getSmoothness() === -1) {
                    this.setSerieVisibility(series.learn__smoothed__, false);
                    this.setSerieVisibility(series.test__smoothed__, false);
                }

                if (this.layoutDisabled['learn']) {
                    this.setSerieVisibility(series.learn, false);
                    this.setSerieVisibility(series.learn__smoothed__, false);
                }

                if (this.layoutDisabled['test']) {
                    this.setSerieVisibility(series.test, false);
                    this.setSerieVisibility(series.test__smoothed__, false);
                    this.setSerieVisibility(series.test__min__, false);
                }
            }
        }
    }
};

CatboostIpython.prototype.getSmoothness = function() {
    return this.smoothness && this.smoothness > -1 ? this.smoothness : -1;
};

CatboostIpython.prototype.setSmoothness = function(weight) {
    if (weight < 0 && weight !== -1 || weight > 1) {
        return;
    }

    this.smoothness = weight;
};

CatboostIpython.prototype.calcSmoothSeries = function() {
    var seriesHash = this.groupSeries(),
        serie,
        smoothedSerie,
        colorFlag,
        enabled = this.getSmoothness() > -1;

    for (var name in seriesHash) {
        if (seriesHash.hasOwnProperty(name)) {
            colorFlag = false;
            serie = seriesHash[name].series.learn;
            smoothedSerie = seriesHash[name].series.learn__smoothed__;

            if (serie && serie.visible) {
                if (enabled) {
                    this.smoothSeries(serie.y, smoothedSerie);
                    colorFlag = true;
                }

                this.highlightSmoothSeries(serie, smoothedSerie, colorFlag);
            }

            colorFlag = false;
            serie = seriesHash[name].series.test;
            smoothedSerie = seriesHash[name].series.test__smoothed__;

            if (serie && serie.visible) {
                if (enabled) {
                    this.smoothSeries(serie.y, smoothedSerie);
                    colorFlag = true;
                }

                this.highlightSmoothSeries(serie, smoothedSerie, colorFlag);
            }
        }
    }
};

CatboostIpython.prototype.highlightSmoothSeries = function(serie, smoothedSerie, flag) {
    if (flag) {
        smoothedSerie.line.color = serie.line._initial_color;
        serie.line.color = smoothedSerie.line._initial_color;
        serie.hoverinfo = 'skip';
    } else {
        serie.line.color = serie.line._initial_color;
        serie.hoverinfo = 'text+x';
    }
};

CatboostIpython.prototype.smoothSeries = function(data, serie) {
    var smoothedPoints = this.smooth(data, this.getSmoothness());

    data.forEach(function(d, index) {
        serie.y[index] = smoothedPoints[index];
        serie.hovertext[index] = smoothedPoints[index].toPrecision(7);
    });
};

CatboostIpython.prototype.redrawActiveChart = function() {
    this.chartsToRedraw[this.activeTab] = true;

    this.redrawAll();
};

CatboostIpython.prototype.redraw = function() {
    if (this.chartsToRedraw[this.activeTab]) {
        this.chartsToRedraw[this.activeTab] = false;

        this.updateSeriesVisibility();
        this.updateSeriesMin();
        this.updateSeriesValues();

        this.calcSmoothSeries();

        this.plotly.redraw(this.traces[this.activeTab].parent);
    }

    this.drawSeries();
};

CatboostIpython.prototype.addRedrawFunc = function() {
    this.redrawFunc = throttle(this.redraw, 400, false, this);
};

CatboostIpython.prototype.redrawAll = function() {
    if (!this.redrawFunc) {
        this.addRedrawFunc();
    }

    this.redrawFunc();
};

CatboostIpython.prototype.addPoints = function(parent, data, type) {
    var iterIndex = 0,
        self = this;

    data.fields.forEach(function(name, index) {
        if (name === 'iter') {
            iterIndex = index;
        }
    });

    data.fields.forEach(function(name, index) {
        if (name === 'iter') {
            return;
        }

        var params = {chartName: name, index: index, train: data.train, type: type, path: data.path},
            key = self.getChartKey(params);

        if (!self.activeTab) {
            self.activeTab = key.chartId;
        }

        var trace = self.getTrace(parent, params),
            smoothedTrace = self.getTrace(parent, $.extend({smoothed: true}, params));

        if (type === 'test') {
            self.getTrace(parent, $.extend({min: true}, params));
        }

        data.chunks.forEach(function(value) {
            if (typeof value[index] === 'undefined') {
                return;
            }

            var pointIndex = value[iterIndex];

            trace.x[pointIndex] = pointIndex;
            trace.y[pointIndex] = value[index];
            trace.hovertext[pointIndex] = value[index].toPrecision(7);

            smoothedTrace.x[pointIndex] = value[iterIndex];
        });

        self.chartsToRedraw[key.chartId] = true;

        self.redrawAll();
    });
};

CatboostIpython.prototype.getChartNode = function(params, active) {
    var node = $('<div class="catboost-graph__chart" tabid="' + params.id + '"></div>');

    if (active) {
        node.addClass('catboost-graph__chart_active');
    }

    return node;
};

CatboostIpython.prototype.getChartTab = function(params, active) {
    var node = $('<div class="catboost-graph__tab" tabid="' + params.id + '">' + params.name + '</div>');

    if (active) {
        node.addClass('catboost-graph__tab_active');
    }

    return node;
};

CatboostIpython.prototype.forEveryChart = function(callback) {
    for (var name in this.traces) {
        if (this.traces.hasOwnProperty(name)) {
            callback(this.traces[name]);
        }
    }
};

CatboostIpython.prototype.forEveryLayout = function(callback) {
    this.forEveryChart(function(chart) {
        callback(chart.layout);
    });
};

CatboostIpython.prototype.getChart = function(parent, params) {
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
                range: [0, Number(this.meta[params.path].iterCount)],
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
            margin: {l: 35, r: 0, t: 35, b: 30},
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

    chartNode[0].on('plotly_hover', function(e) {
        self.updateSeriesValues(e.points[0].x);
    });

    chartNode[0].on('plotly_click', function(e) {
        self.updateSeriesValues(e.points[0].x, true);
    });

    return this.charts[id];
};


CatboostIpython.prototype.getTrace = function(parent, params) {
    var key = this.getChartKey(params),
        chartSeries = [];

    if (this.traces[key.chartId]) {
        chartSeries = this.traces[key.chartId].traces.filter(function(trace) {
            return trace.name === key.seriesId;
        });
    }

    if (chartSeries.length) {
        return chartSeries[0];
    } else {
        this.getChart(parent, {id: key.chartId, name: params.chartName, path: params.path});

        var color = this.getNextColor(params.path, params.smoothed ? 0.1 : 1),
            trace = {
                name: key.seriesId,
                _params: params,
                x: [],
                y: [],
                hovertext: [],
                hoverinfo: 'text+x',
                line: {
                    width: 1,
                    dash: params.type === 'test' ? 'solid' : 'dot',
                    color: color,
                    _initial_color: color
                },
                mode: 'lines',
                hoveron: 'points'
            };

        if (params.min) {
            trace = {
                name: key.seriesId,
                _params: params,
                x: [],
                y: [],
                marker: {
                    width: 2,
                    color: color,
                    _initial_color: color
                },
                hovertext: [],
                hoverinfo: 'text',
                mode: 'markers',
                type: 'scatter'
            };
        }

        this.traces[key.chartId].traces.push(trace);

        return trace;
    }
};

CatboostIpython.prototype.getChartKey = function(params) {
    return {
        chartId: params.chartName + ' ' + params.index,
        seriesId: params.train + ' ' + params.type + (params.min ? '__min__' : '') + (params.smoothed ? '__smoothed__' : ''),
        colorId: params.train
    };
};

CatboostIpython.prototype.cleanSeries = function() {
    $('.catboost-panel__series', this.layout).html('');
};

CatboostIpython.prototype.groupSeries = function() {
    var series = this.traces[this.activeTab].traces,
        index = 0,
        seriesHash = {};

    series.map(function(serie) {
        var name = serie._params.train,
            prefix = serie._params.type;

        if (serie._params.min) {
            prefix += '__min__';
        }

        if (serie._params.smoothed) {
            prefix += '__smoothed__';
        }

        if (!seriesHash[name]) {
            seriesHash[name] = {
                index: index,
                series: {}
            };

            index++;
        }

        seriesHash[name].series[prefix] = serie;
    });

    return seriesHash;
};

CatboostIpython.prototype.drawSeries = function() {
    if ($('.catboost-panel__series .catboost-panel__serie', this.layout).length) {
        return;
    }

    var html = '',
        seriesHash = this.groupSeries();

    for (var name in seriesHash) {
        if (seriesHash.hasOwnProperty(name)) {
            html += this.drawSerie(name, seriesHash[name]);
        }
    }

    $('.catboost-panel__series', this.layout).html(html);

    this.updateSeriesValues();

    this.addSeriesEvents();
};

CatboostIpython.prototype.drawSerie = function(name, hash) {
    var id = 'catboost-serie-' + this.index + '-' + hash.index,
        html = '<div id="' + id + '" class="catboost-panel__serie" style="color:' + hash.series.learn.line._initial_color + '">' +
                    '<div class="catboost-panel__serie_top">' +
                        '<input type="checkbox" data-seriename="' + name + '" class="catboost-panel__serie_checkbox" id="' + id + '-box" ' + (!this.layoutDisabled.series[name] ? 'checked="checked"' : '') + '></input>' +
                        '<label title=' + this.meta[hash.series.learn._params.path].name + ' for="' + id + '-box" class="catboost-panel__serie_label">' + name + '<div class="catboost-panel__serie_time_left" title="Estimate time"></div></label>' +
                        '<div class="catboost-panel__serie_time">' +
                            '<div class="catboost-panel__serie_time_spend" title="Time spend"></div>' +
                        '</div>' +
                    '</div>' +
                    '<div class="catboost-panel__serie_middle catboost-panel__serie__learn_hint">' +
                        '<div class="catboost-panel__serie_hint">curr</div>' +
                        '<div class="catboost-panel__serie_learn_pic" style="border-color:' + hash.series.learn.line._initial_color + '"></div>' +
                        '<div class="catboost-panel__serie_learn_value"></div>' +
                        '<div class="catboost-panel__serie_test_pic" style="border-color:' + hash.series.learn.line._initial_color + '"></div>' +
                        '<div class="catboost-panel__serie_test_value"></div>' +
                        '<div class="catboost-panel__serie_iteration" title="curr iteration"></div>' +
                    '</div>' +
                    '<div class="catboost-panel__serie_bottom">' +
                        '<div class="catboost-panel__serie_hint catboost-panel__serie__test_hint">best</div>' +
                        '<div class="catboost-panel__serie_learn_pic" style="border-color:transparent"></div>' +
                        '<div class="catboost-panel__serie_best_learn_value"></div>' +
                        '<div class="catboost-panel__serie_test_pic"></div>' +
                        '<div class="catboost-panel__serie_best_test_value"></div>' +
                        '<div class="catboost-panel__serie_best_iteration" title="best iteration"></div>' +
                    '</div>' +
                '</div>';

    return html;
};

CatboostIpython.prototype.updateSeriesValues = function(iteration, click) {
    var seriesHash = this.groupSeries();

    for (var name in seriesHash) {
        if (seriesHash.hasOwnProperty(name) && !this.layoutDisabled.series[name]) {
            this.updateSerieValues(name, seriesHash[name], iteration, click);
        }
    }
};

CatboostIpython.prototype.updateSeriesMin = function() {
    var seriesHash = this.groupSeries();

    for (var name in seriesHash) {
        if (seriesHash.hasOwnProperty(name) && !this.layoutDisabled.series[name]) {
            this.updateSerieMin(name, seriesHash[name]);
        }
    }
};

CatboostIpython.prototype.getBestValue = function(data, path) {
    if (!data.length) {
        return {
            best: undefined,
            index: -1
        };
    }

    var best = data[0],
        index = 0,
        func = this.meta[path]['loss_' + this.traces[this.activeTab].name];

    for (var i = 1, l = data.length; i < l; i++) {
        if (func === 'min' && data[i] < best) {
            best = data[i];
            index = i;
        }

        if (func === 'max' && data[i] > best) {
            best = data[i];
            index = i;
        }
    }

    return {
        best: best,
        index: index,
        func: func
    };
};

CatboostIpython.prototype.formatItemValue = function(value, index, suffix) {
    if (typeof value === 'undefined') {
        return '';
    }

    suffix = suffix || '';

    return '<span title="' + suffix + 'value ' + value + '">' + value + '</span>';
};

CatboostIpython.prototype.updateSerieMin = function(name, hash) {
    if (!(hash.series.test && hash.series.test__min__)) {
        return;
    }

    var testData = hash.series.test.y,
        path = this.getSeriesPath(hash),
        testBestValue = this.getBestValue(testData, path);

    if (testBestValue.index === -1) {
        return;
    }

    hash.series.test__min__.x[0] = testBestValue.index;
    hash.series.test__min__.y[0] = testBestValue.best;
    hash.series.test__min__.hovertext[0] = testBestValue.func + ': ' + testBestValue.index + ' ' + testBestValue.best;
};

CatboostIpython.prototype.getSeriesPath = function(hash) {
    if (hash.series.test) {
        return hash.series.test._params.path;
    }

    if (hash.series.learn) {
        return hash.series.learn._params.path;
    }
};

CatboostIpython.prototype.updateSerieValues = function(name, hash, iteration, click) {
    var id = 'catboost-serie-' + this.index + '-' + hash.index,
        learn = hash.series.learn,
        learnData = learn ? learn.y : [],
        test = hash.series.test,
        testData = test ? test.y : [],
        index = typeof iteration !== 'undefined' && iteration < learnData.length - 1 ? iteration : learnData.length - 1,
        learnValue = learnData.length ? learnData[index] : undefined,
        testValue = testData.length ? testData[index] : undefined,
        path = this.getSeriesPath(hash),
        testBestValue = this.getBestValue(testData, path),
        timeLeft = '',
        timeSpend = '';

    if (click || !this.clickMode) {
        $('#' + id + ' .catboost-panel__serie_learn_value', this.layout).html(this.formatItemValue(learnValue, index, 'learn '));
        $('#' + id + ' .catboost-panel__serie_test_value', this.layout).html(this.formatItemValue(testValue, index, 'test '));
        $('#' + id + ' .catboost-panel__serie_iteration', this.layout).html(index);

        if (this.timeLeft[path][learnData.length - 1]) {
            timeLeft = Math.ceil(Number(this.timeLeft[path][learnData.length - 1][1]) / 1000) * 1000;
        }
        $('#' + id + ' .catboost-panel__serie_time_left', this.layout).html(timeLeft ? ('~' + this.convertTime(timeLeft)) : '');

        if (this.timeLeft[path][index]) {
            timeSpend = Math.ceil(Number(this.timeLeft[path][index][2]) / 1000) * 1000;
        }

        $('#' + id + ' .catboost-panel__serie_time_spend', this.layout).html(this.convertTime(timeSpend));
        $('#' + id + ' .catboost-panel__serie_best_iteration', this.layout).html(testBestValue.index > -1 ? testBestValue.index : '');


        $('#' + id + ' .catboost-panel__serie_best_test_value', this.layout).html(this.formatItemValue(testBestValue.best, testBestValue.index, 'best test '));
    }

    if (click) {
        this.clickMode = true;

        $('#catboost-control2-clickmode' + this.index, this.layout)[0].checked = true;
    }
};

CatboostIpython.prototype.addSeriesEvents = function() {
    var self = this;

    $('.catboost-panel__serie_checkbox', this.layout).click(function() {
        var name = $(this).data('seriename');

        self.layoutDisabled.series[name] = !$(this)[0].checked;

        self.redrawActiveChart();
    });
};

CatboostIpython.prototype.getNextColor = function(path, opacity) {
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

CatboostIpython.prototype.hexToRgba = function(value, opacity) {
    if (value.length < 6) {
        var pattern = /^#?([a-f\d])([a-f\d])([a-f\d])/i;
        value = value.replace(pattern, function(m, r, g, b) {
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

CatboostIpython.prototype.convertTime = function(time) {
    if (!time) {
        return '0s';
    }

    var millis = time % 1000;
    time = parseInt(time / 1000, 10);
    var seconds = time % 60;
    time = parseInt(time / 60, 10);
    var minutes = time % 60;
    time = parseInt(time / 60, 10);
    var hours = time % 24;
    var out = "";
    if (hours && hours > 0) {
        out += hours + 'h ';
    }
    if (minutes && minutes > 0) {
        out += minutes + 'm ';
    }
    if (seconds && seconds > 0) {
        out += seconds + 's ';
    }
    if (millis && millis > 0) {
        out += millis + 'ms';
    }

    return out.trim();
};

CatboostIpython.prototype.mean = function(values, valueof) {
    var n = values.length,
        m = n,
        i = -1,
        value,
        sum = 0,
        number = function(x) {
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
CatboostIpython.prototype.smooth = function(data, weight) {
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
            res.push(self.mean(data.slice(start, end).filter(function(d) {
                return isFinite(d);
            }), null));
        }
    });

    return res;
};
