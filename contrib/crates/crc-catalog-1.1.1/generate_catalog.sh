#!/bin/sh

echo "use crate::Algorithm;"
curl -s https://reveng.sourceforge.io/crc-catalogue/all.htm | grep -o 'width.*name.*"' | while read -r line; do
  # echo $(echo $line | \
  #   sed 's/ /, /g' | \
  #   sed 's/[-\/]/_/g' | \
  #   sed 's/width=\([0-9]*\), \(.*\), name="\(.*\)"/pub const \3: Algorithm<u\1> = Algorithm { \2 };/')
  
  width=$(echo $line | sed 's/width=\([0-9]*\) \(.*\) name="\(.*\)"/\1/')
  params=$(echo $line | sed 's/width=\([0-9]*\) \(.*\) name="\(.*\)"/\2/' | sed 's/ /, /g' | sed 's/=/: /g')
  name=$(echo $line | sed 's/width=\([0-9]*\) \(.*\) name="\(.*\)"/\3/' | sed 's/[-\/]/_/g')
  if [ $width -eq 8 ] || [ $width -eq 16 ] || [ $width -eq 32 ] || [ $width -eq 64 ]; then
    echo "pub const $name: Algorithm<u$width> = Algorithm { $params };"
  fi
done
