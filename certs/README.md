# CA certificates

`cacert.pem` contains the common public root CAs and the internal CAs from
`https://crls.yandex.net/allCAs.pem`. The `certs` library embeds this bundle as
the `/builtin/cacert` resource. `update-certs.py` updates both `cacert.pem` and
the internal-only `yandex_internal.pem` bundle.

## C++ clients in self-contained deployments

Arcadia's Linux libcurl uses `/etc/ssl/certs/ca-certificates.crt` and
`/etc/ssl/certs` by default. Minimal containers and other self-contained
runtime environments may not provide those paths. In that case an HTTPS
request fails with curl error 77:

```text
Problem with the SSL CA cert (path? access rights?)
```

Error 77 means that libcurl could not load its CA file or directory. It occurs
before validation of the server certificate. Do not work around it by setting
`CURLOPT_SSL_VERIFYPEER` or `CURLOPT_SSL_VERIFYHOST` to zero.

To make an Arcadia C++ binary independent of the runtime filesystem, add
`certs` and `library/cpp/resource` to `PEERDIR`, then pass the embedded bundle
to libcurl:

```cpp
#include <library/cpp/resource/resource.h>

static const auto caInfo = NResource::Find("/builtin/cacert");
const curl_blob caInfoBlob{
    const_cast<char*>(caInfo.data()),
    caInfo.size(),
    CURL_BLOB_NOCOPY};

curl_easy_setopt(curl, CURLOPT_CAINFO_BLOB, &caInfoBlob);
```

The resource must outlive every request that uses `CURL_BLOB_NOCOPY`; a static
value provides that lifetime. Use `CURL_BLOB_COPY` if the source data has a
shorter lifetime. Check the return value of `curl_easy_setopt` in production
code.

If the deployment intentionally maintains an OS CA store, keeping libcurl's
defaults is also valid. Verify that the compiled paths above exist and are
readable in the actual workload container, not only on the build host.
