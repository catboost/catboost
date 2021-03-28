__webpack_public_path__ = document.querySelector('body').getAttribute('data-base-url') + 'nbextensions/hello_world_widget';

if (window.require) {
    window.require.config({
        map: {
            "*": {
                "catboost_widget": "nbextensions/catboost_widget/index",
            }
        }
    });
}

module.exports = {
    load_ipython_extension: function () {
    }
};
