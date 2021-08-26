# Additional packages for data visualization support

Execute the following steps to support the data visualization feature in Jupyter Notebook:
1. Install the `ipywidgets`{{ python-package }} (version 7.x or higher is required):
    
    ```no-highlight
    pip install ipywidgets
    ```
    
    {% note warning %}
    
    The visualization of previously created documents does not work after updating to ipywidgets 7.x. Perform the following steps to make the old contents work:
    1. Create a new Jupyter document. The file name must differ from the old one.
    1. Paste the contents of the old file to the new one.
    1. (Optionally) Refresh the Notebook page and restart the kernel if the visualization does not work.
    
    {% endnote %}
    
1. Turn on the widgets extension:
    
    ```no-highlight
    jupyter nbextension enable --py widgetsnbextension
    ```

