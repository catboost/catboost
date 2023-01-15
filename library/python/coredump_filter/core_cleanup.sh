#!/usr/bin/env bash
srv="http://arachnid09.search.yandex.net:37771"

for core_hash in $( curl "$srv/show?server=&ctype=&service=" | /bin/grep 'button.*id'| perl -p -e 's/.*id="(.*?)".*/\1/' ) ; do
    curl "$srv/fixed/$core_hash/"
    echo "Cleaned $core_hash"
done
