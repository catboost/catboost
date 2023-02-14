import logging
import six
import typing as tp


def detect_recursive_dict(obj):
    found_recursive = False
    for _, _ in _detect_recursive_dict(obj):
        found_recursive = True

    return found_recursive


def _detect_recursive_dict(obj):
    # type: (dict) -> tp.Iterable[tuple[str, str]]

    assert isinstance(obj, dict)

    logger = logging.getLogger("recursive_detector")

    logger.info("Start detecting recursive values in %s", id(obj))

    id_keys = dict()
    ids = set()

    processing = [("[root]", obj)]

    while processing:
        k, v = processing.pop()

        if isinstance(v, (dict, list, set, tuple)):
            value_id = id(v)

            if value_id in ids:
                found_key = id_keys[value_id]
                if k.startswith(found_key):
                    if isinstance(v, dict):
                        additional_info = "keys: {}".format(v.keys())
                    else:
                        additional_info = "items: {}".format(len(v))

                    logger.warning(
                        "FOUND RECURSIVE: hash `%r`\nKey `%s` vs `%s`\nValue id: %s\ntype: %s\nAdditional info: `%s`",
                        value_id,
                        found_key,
                        k,
                        id(v),
                        type(v),
                        additional_info,
                    )

                    yield found_key, k

            elif isinstance(v, dict):
                id_keys[value_id] = k
                ids.add(value_id)

                for sub_key, sub_value in six.iteritems(v):
                    processing.append(("{}.{}".format(k, sub_key), sub_value))

            elif isinstance(v, (list, tuple, set)):
                id_keys[value_id] = v
                ids.add(value_id)

                for index, sub_value in enumerate(tuple(v)):
                    sub_key = "{}.{}#{}".format(k, type(v).__name__, index)
                    processing.append((sub_key, sub_value))
