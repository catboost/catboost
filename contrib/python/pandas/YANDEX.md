# How to update

Download Pandas sources zip from https://pandas.pydata.org/ and run in this directory:

```sh
unzip pandas-0.19.1.zip
rm -rf pandas
mv pandas-0.19.1/pandas .
rm -rf pandas-0.19.1
./YANDEX.prepare.sh
./YANDEX.generate.sh
```
