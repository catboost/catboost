var path = require('path');
var version = require('./package.json').version;

var rules = [
    {test: /\.css$/, use: ['style-loader', 'css-loader']}
];


module.exports = [
    {
        entry: './src/extension.js',
        output: {
            filename: 'extension.js',
            path: path.resolve(__dirname, 'nbextension'),
            libraryTarget: 'amd',
            publicPath: ''
        },
    },
    {
        entry: './src/index.js',
        output: {
            filename: 'index.js',
            path: path.resolve(__dirname, 'nbextension'),
            libraryTarget: 'amd',
            publicPath: '',
        },
        module: {
            rules: rules
        },
        externals: ['@jupyter-widgets/base']
    },
    {
        entry: './src/embed.js',
        output: {
            filename: 'index.js',
            path: path.resolve(__dirname, 'dist'),
            libraryTarget: 'amd',
            publicPath: 'https://unpkg.com/catboost-widget@' + version + '/dist/'
        },
        module: {
            rules: rules
        },
        externals: ['@jupyter-widgets/base']
    }
];
