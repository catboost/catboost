def get_include_callback():
    """
    .. function: get_include_callback    returns function that processes each DOM element to get xsyn include from it, and it's aware of directory with all the xsyns.

        :param  xsyn_dir    directory with xsyns.
    """
    def get_include(element):
        """
        .. function:    get_include     returns list of includes from this DOM element.

            :param  element     DOM element.
        """
        res = []
        if element.nodeType == element.ELEMENT_NODE and element.nodeName == "parse:include":
            attrs = element.attributes
            for i in xrange(attrs.length):
                attr = attrs.item(i)
                if attr.nodeName == "path":
                    include_filename = attr.nodeValue
                    res.append(include_filename)
        return res

    return get_include


def traverse_xsyn(element, on_element):
    """
    .. function: traverse_xsyn  traverses element and returns concatenated lists of calling on_element of each element.

        :param  element     element in DOM.
        :param  on_element  callback on element that returns list of values.
    """
    res = on_element(element)
    for child in element.childNodes:
        child_results = traverse_xsyn(child, on_element)
        res += child_results
    return res


def process_xsyn(filepath, on_element):
    """
    .. function:    process_xsyn    processes xsyn file and return concatenated list of calling on_element on each DOM element.

        :param  filepath    path to xsyn file
        :param  on_element  callback called on each element in xsyn that returns list of values.

    """

    # keep a stack of filepathes if on_element calls process_xsyn recursively
    with open(filepath) as xsyn_file:
        from xml.dom.minidom import parse
        tree = parse(xsyn_file)
        tree.normalize()
        res = traverse_xsyn(tree, on_element)
    return res


def get_all_includes(filepath):
    callback = get_include_callback()
    return process_xsyn(filepath, callback)
