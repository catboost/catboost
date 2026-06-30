
{{ product }} interprets the value of a numerical feature as a missing value if it is equal to one of the following values:

- `None`

- [Floating point NaN value](https://en.wikipedia.org/wiki/NaN)

- One of the following strings when loading the values from files or as Python strings:

    , <q>#N/A</q>, <q>#N/A N/A</q>, <q>#NA</q>, <q>-1.#IND</q>, <q>-1.#QNAN</q>, <q>-NaN</q>, <q>-nan</q>, <q>1.#IND</q>, <q>1.#QNAN</q>, <q>N/A</q>, <q>NA</q>, <q>NULL</q>, <q>NaN</q>, <q>n/a</q>, <q>nan</q>, <q>null</q>, <q>NAN</q>, <q>Na</q>, <q>na</q>, <q>Null</q>, <q>none</q>, <q>None</q>, <q>-</q>

    This is an extended version of the default missing values list in [pandas](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html).

