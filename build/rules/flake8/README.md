#Flake8 migrations

##migrations.yaml
Format:
```
migrations:
  plugin-1:
    ignore:
      - B102
      - S103
      - F401
    prefixes:
      - devtools/ya
      - ads
      - quality
  ignore-F123:
    ignore:
      - F123
    prefixes:
      - devtools/ya
      - devtools/d
```
If arcadia-relative filepath startswith prefix from prefixes, then:

1. ignore values will be added to flake8.conf ignore section

