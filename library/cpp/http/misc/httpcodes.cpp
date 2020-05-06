#include "httpcodes.h"

TStringBuf HttpCodeStrEx(int code) noexcept {
    switch (code) {
        case HTTP_CONTINUE:
            return AsStringBuf("100 Continue");
        case HTTP_SWITCHING_PROTOCOLS:
            return AsStringBuf("101 Switching protocols");
        case HTTP_PROCESSING:
            return AsStringBuf("102 Processing");

        case HTTP_OK:
            return AsStringBuf("200 Ok");
        case HTTP_CREATED:
            return AsStringBuf("201 Created");
        case HTTP_ACCEPTED:
            return AsStringBuf("202 Accepted");
        case HTTP_NON_AUTHORITATIVE_INFORMATION:
            return AsStringBuf("203 None authoritative information");
        case HTTP_NO_CONTENT:
            return AsStringBuf("204 No content");
        case HTTP_RESET_CONTENT:
            return AsStringBuf("205 Reset content");
        case HTTP_PARTIAL_CONTENT:
            return AsStringBuf("206 Partial content");
        case HTTP_MULTI_STATUS:
            return AsStringBuf("207 Multi status");
        case HTTP_ALREADY_REPORTED:
            return AsStringBuf("208 Already reported");
        case HTTP_IM_USED:
            return AsStringBuf("226 IM used");

        case HTTP_MULTIPLE_CHOICES:
            return AsStringBuf("300 Multiple choices");
        case HTTP_MOVED_PERMANENTLY:
            return AsStringBuf("301 Moved permanently");
        case HTTP_FOUND:
            return AsStringBuf("302 Moved temporarily");
        case HTTP_SEE_OTHER:
            return AsStringBuf("303 See other");
        case HTTP_NOT_MODIFIED:
            return AsStringBuf("304 Not modified");
        case HTTP_USE_PROXY:
            return AsStringBuf("305 Use proxy");
        case HTTP_TEMPORARY_REDIRECT:
            return AsStringBuf("307 Temporarily redirect");
        case HTTP_PERMANENT_REDIRECT:
            return AsStringBuf("308 Permanent redirect");

        case HTTP_BAD_REQUEST:
            return AsStringBuf("400 Bad request");
        case HTTP_UNAUTHORIZED:
            return AsStringBuf("401 Unauthorized");
        case HTTP_PAYMENT_REQUIRED:
            return AsStringBuf("402 Payment required");
        case HTTP_FORBIDDEN:
            return AsStringBuf("403 Forbidden");
        case HTTP_NOT_FOUND:
            return AsStringBuf("404 Not found");
        case HTTP_METHOD_NOT_ALLOWED:
            return AsStringBuf("405 Method not allowed");
        case HTTP_NOT_ACCEPTABLE:
            return AsStringBuf("406 Not acceptable");
        case HTTP_PROXY_AUTHENTICATION_REQUIRED:
            return AsStringBuf("407 Proxy Authentication required");
        case HTTP_REQUEST_TIME_OUT:
            return AsStringBuf("408 Request time out");
        case HTTP_CONFLICT:
            return AsStringBuf("409 Conflict");
        case HTTP_GONE:
            return AsStringBuf("410 Gone");
        case HTTP_LENGTH_REQUIRED:
            return AsStringBuf("411 Length required");
        case HTTP_PRECONDITION_FAILED:
            return AsStringBuf("412 Precondition failed");
        case HTTP_REQUEST_ENTITY_TOO_LARGE:
            return AsStringBuf("413 Request entity too large");
        case HTTP_REQUEST_URI_TOO_LARGE:
            return AsStringBuf("414 Request uri too large");
        case HTTP_UNSUPPORTED_MEDIA_TYPE:
            return AsStringBuf("415 Unsupported media type");
        case HTTP_REQUESTED_RANGE_NOT_SATISFIABLE:
            return AsStringBuf("416 Requested Range Not Satisfiable");
        case HTTP_EXPECTATION_FAILED:
            return AsStringBuf("417 Expectation Failed");
        case HTTP_I_AM_A_TEAPOT:
            return AsStringBuf("418 I Am A Teapot");
        case HTTP_AUTHENTICATION_TIMEOUT:
            return AsStringBuf("419 Authentication Timeout");
        case HTTP_MISDIRECTED_REQUEST:
            return AsStringBuf("421 Misdirected Request");
        case HTTP_UNPROCESSABLE_ENTITY:
            return AsStringBuf("422 Unprocessable Entity");
        case HTTP_LOCKED:
            return AsStringBuf("423 Locked");
        case HTTP_FAILED_DEPENDENCY:
            return AsStringBuf("424 Failed Dependency");
        case HTTP_UPGRADE_REQUIRED:
            return AsStringBuf("426 Upgrade Required");
        case HTTP_PRECONDITION_REQUIRED:
            return AsStringBuf("428 Precondition Required");
        case HTTP_TOO_MANY_REQUESTS:
            return AsStringBuf("429 Too Many Requests");
        case HTTP_UNAVAILABLE_FOR_LEGAL_REASONS:
            return AsStringBuf("451 Unavailable For Legal Reason");

        case HTTP_INTERNAL_SERVER_ERROR:
            return AsStringBuf("500 Internal server error");
        case HTTP_NOT_IMPLEMENTED:
            return AsStringBuf("501 Not implemented");
        case HTTP_BAD_GATEWAY:
            return AsStringBuf("502 Bad gateway");
        case HTTP_SERVICE_UNAVAILABLE:
            return AsStringBuf("503 Service unavailable");
        case HTTP_GATEWAY_TIME_OUT:
            return AsStringBuf("504 Gateway time out");
        case HTTP_HTTP_VERSION_NOT_SUPPORTED:
            return AsStringBuf("505 HTTP version not supported");
        case HTTP_VARIANT_ALSO_NEGOTIATES:
            return AsStringBuf("506 Variant also negotiates");
        case HTTP_INSUFFICIENT_STORAGE:
            return AsStringBuf("507 Insufficient storage");
        case HTTP_LOOP_DETECTED:
            return AsStringBuf("508 Loop Detected");
        case HTTP_BANDWIDTH_LIMIT_EXCEEDED:
            return AsStringBuf("509 Bandwidth Limit Exceeded");
        case HTTP_NOT_EXTENDED:
            return AsStringBuf("510 Not Extended");
        case HTTP_NETWORK_AUTHENTICATION_REQUIRED:
            return AsStringBuf("511 Network Authentication Required");
        case HTTP_UNASSIGNED_512:
            return AsStringBuf("512 Unassigned");

        default:
            return AsStringBuf("000 Unknown HTTP code");
    }
}
