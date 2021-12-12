# Distributed Learning

To train in distibuted mode follow these steps:

1. Start a worker with `catboost run-worker --node-port {port}` on each worker host. Any free port is ok. Ports do not have to be equal for all workers.
1. Create plain text file containing hostnames. Each line of the file should contains one line per worker: `hostname:port`

    For example:

    ```
    192.168.1.1:9999
    192.168.1.2:9999
    ```

    If host names are IPv6 addresses they must be in square brackets, for example `[2001:0db8:85a3:0000:0000:8a2e:0370:7334]`

1. Start {{ product }} on the main host with the regular training options, adding `--node-type Master --file-with-hosts {filename}`.
