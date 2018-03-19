#include "httpcodes.h"

TStringBuf HttpCodeStrEx(int code) noexcept {
    switch (code) {
        case HTTP_CONTINUE:
            return STRINGBUF("100 Continue");
        case HTTP_SWITCHING_PROTOCOLS:
            return STRINGBUF("101 Switching protocols");
        case HTTP_PROCESSING:
            return STRINGBUF("102 Processing");

        case HTTP_OK:
            return STRINGBUF("200 Ok");
        case HTTP_CREATED:
            return STRINGBUF("201 Created");
        case HTTP_ACCEPTED:
            return STRINGBUF("202 Accepted");
        case HTTP_NON_AUTHORITATIVE_INFORMATION:
            return STRINGBUF("203 None authoritative information");
        case HTTP_NO_CONTENT:
            return STRINGBUF("204 No content");
        case HTTP_RESET_CONTENT:
            return STRINGBUF("205 Reset content");
        case HTTP_PARTIAL_CONTENT:
            return STRINGBUF("206 Partial content");
        case HTTP_MULTI_STATUS:
            return STRINGBUF("207 Multi status");
        case HTTP_ALREADY_REPORTED:
            return STRINGBUF("208 Already reported");
        case HTTP_IM_USED:
            return STRINGBUF("226 IM used");

        case HTTP_MULTIPLE_CHOICES:
            return STRINGBUF("300 Multiple choices");
        case HTTP_MOVED_PERMANENTLY:
            return STRINGBUF("301 Moved permanently");
        case HTTP_FOUND:
            return STRINGBUF("302 Moved temporarily");
        case HTTP_SEE_OTHER:
            return STRINGBUF("303 See other");
        case HTTP_NOT_MODIFIED:
            return STRINGBUF("304 Not modified");
        case HTTP_USE_PROXY:
            return STRINGBUF("305 Use proxy");
        case HTTP_TEMPORARY_REDIRECT:
            return STRINGBUF("307 Temporarily redirect");
        case HTTP_PERMANENT_REDIRECT:
            return STRINGBUF("308 Permanent redirect");

        case HTTP_BAD_REQUEST:
            return STRINGBUF("400 Bad request");
        case HTTP_UNAUTHORIZED:
            return STRINGBUF("401 Unauthorized");
        case HTTP_PAYMENT_REQUIRED:
            return STRINGBUF("402 Payment required");
        case HTTP_FORBIDDEN:
            return STRINGBUF("403 Forbidden");
        case HTTP_NOT_FOUND:
            return STRINGBUF("404 Not found");
        case HTTP_METHOD_NOT_ALLOWED:
            return STRINGBUF("405 Method not allowed");
        case HTTP_NOT_ACCEPTABLE:
            return STRINGBUF("406 Not acceptable");
        case HTTP_PROXY_AUTHENTICATION_REQUIRED:
            return STRINGBUF("407 Proxy Authentication required");
        case HTTP_REQUEST_TIME_OUT:
            return STRINGBUF("408 Request time out");
        case HTTP_CONFLICT:
            return STRINGBUF("409 Conflict");
        case HTTP_GONE:
            return STRINGBUF("410 Gone");
        case HTTP_LENGTH_REQUIRED:
            return STRINGBUF("411 Length required");
        case HTTP_PRECONDITION_FAILED:
            return STRINGBUF("412 Precondition failed");
        case HTTP_REQUEST_ENTITY_TOO_LARGE:
            return STRINGBUF("413 Request entity too large");
        case HTTP_REQUEST_URI_TOO_LARGE:
            return STRINGBUF("414 Request uri too large");
        case HTTP_UNSUPPORTED_MEDIA_TYPE:
            return STRINGBUF("415 Unsupported media type");
        case HTTP_REQUESTED_RANGE_NOT_SATISFIABLE:
            return STRINGBUF("416 Requested Range Not Satisfiable");
        case HTTP_EXPECTATION_FAILED:
            return STRINGBUF("417 Expectation Failed");
        case HTTP_I_AM_A_TEAPOT:
            return STRINGBUF("418 I Am A Teapot");
        case HTTP_AUTHENTICATION_TIMEOUT:
            return STRINGBUF("419 Authentication Timeout");
        case HTTP_MISDIRECTED_REQUEST:
            return STRINGBUF("421 Misdirected Request");
        case HTTP_UNPROCESSABLE_ENTITY:
            return STRINGBUF("422 Unprocessable Entity");
        case HTTP_LOCKED:
            return STRINGBUF("423 Locked");
        case HTTP_FAILED_DEPENDENCY:
            return STRINGBUF("424 Failed Dependency");
        case HTTP_UPGRADE_REQUIRED:
            return STRINGBUF("426 Upgrade Required");
        case HTTP_PRECONDITION_REQUIRED:
            return STRINGBUF("428 Precondition Required");
        case HTTP_TOO_MANY_REQUESTS:
            return STRINGBUF("429 Too Many Requests");
        case HTTP_UNAVAILABLE_FOR_LEGAL_REASONS:
            return STRINGBUF("451 Unavailable For Legal Reason");

        case HTTP_INTERNAL_SERVER_ERROR:
            return STRINGBUF("500 Internal server error");
        case HTTP_NOT_IMPLEMENTED:
            return STRINGBUF("501 Not implemented");
        case HTTP_BAD_GATEWAY:
            return STRINGBUF("502 Bad gateway");
        case HTTP_SERVICE_UNAVAILABLE:
            return STRINGBUF("503 Service unavailable");
        case HTTP_GATEWAY_TIME_OUT:
            return STRINGBUF("504 Gateway time out");
        case HTTP_HTTP_VERSION_NOT_SUPPORTED:
            return STRINGBUF("505 HTTP version not supported");
        case HTTP_VARIANT_ALSO_NEGOTIATES:
            return STRINGBUF("506 Variant also negotiates");
        case HTTP_INSUFFICIENT_STORAGE:
            return STRINGBUF("507 Insufficient storage");
        case HTTP_LOOP_DETECTED:
            return STRINGBUF("508 Loop Detected");
        case HTTP_BANDWIDTH_LIMIT_EXCEEDED:
            return STRINGBUF("509 Bandwidth Limit Exceeded");
        case HTTP_NOT_EXTENDED:
            return STRINGBUF("510 Not Extended");
        case HTTP_NETWORK_AUTHENTICATION_REQUIRED:
            return STRINGBUF("511 Network Authentication Required");

        default:
            return STRINGBUF("000 Unknown HTTP code");
    }
}
