#pragma once

namespace NNetlibaSocket {
    namespace NNetliba_v12 {
        const ui8 CMD_POS = 11;
        enum EUdpCmd {
            CMD_BEGIN = 1,

            DATA = CMD_BEGIN,
            DATA_SMALL,   // no jumbo-packets
            DO_NOT_USE_1, //just reserved
            DO_NOT_USE_2, //just reserved

            CANCEL_TRANSFER,

            ACK,
            ACK_COMPLETE,
            ACK_CANCELED,
            ACK_RESEND_NOSHMEM,

            PING,  // 10
            PONG,  // 11
            PONG_IB,

            KILL,

            XS_PING, // 14
            XS_PONG, // 15

            CMD_END,
        };
    }

    namespace NNetliba {
        const ui8 CMD_POS = 8;
        enum EUdpCmd {
            DATA,
            ACK,
            ACK_COMPLETE,
            ACK_RESEND,
            DATA_SMALL, // no jumbo-packets
            PING,
            PONG,
            DATA_SHMEM,
            DATA_SMALL_SHMEM,
            KILL,
            ACK_RESEND_NOSHMEM,
        };
    }

}
