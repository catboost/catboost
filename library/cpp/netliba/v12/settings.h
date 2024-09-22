#pragma once

#include <util/str_stl.h>
#include <util/system/yassert.h>

namespace NNetliba_v12 {
    enum ETos { TOS_DEFAULT = -1 }; // TODO: add more colors
    enum ENetlibaColor { NC_MAXCOLOR = 0xff };

    enum EPacketPriority {
        PP_LOW = 0,
        PP_NORMAL,
        PP_HIGH,
        PP_SYSTEM // It is HIGHEST priority for system
    };

    inline bool IsValidTos(const int tos) {
        return tos == TOS_DEFAULT || 0 <= tos && tos <= 0xFF;
    }

    class TTos {
    public:
        TTos()
            : DataTos(TOS_DEFAULT)
            , AckTos(TOS_DEFAULT)
        {
        }

        int GetDataTos() const {
            return DataTos;
        }
        int GetAckTos() const {
            return AckTos;
        }

        void SetDataTos(const int dataTos) {
            Y_ABORT_UNLESS(IsValidTos(dataTos), "Bad TOS!");
            DataTos = dataTos;
        }
        void SetAckTos(const int ackTos) {
            Y_ABORT_UNLESS(IsValidTos(ackTos), "Bad TOS!");
            AckTos = ackTos;
        }

    private:
        int DataTos;
        int AckTos;
    };

    inline bool operator==(const TTos& lhv, const TTos& rhv) {
        return lhv.GetAckTos() == rhv.GetAckTos() &&
               lhv.GetDataTos() == rhv.GetDataTos();
    }
    inline bool operator!=(const TTos& lhv, const TTos& rhv) {
        return !(lhv == rhv);
    }

    ///////////////////////////////////////////////////////////////////////////////

    class TConnectionSettings {
    public:
        TConnectionSettings()
            : UseTosCongestionAlgo(false)
            , InflateCongestion(false)
            , SmallMtuUseXs(false)
        {
        }

        virtual ~TConnectionSettings() {
        }

        bool GetUseTosCongestionAlgo() const {
            return UseTosCongestionAlgo;
        }
        bool GetInflateCongestion() const {
            return InflateCongestion;
        }
        bool GetSmallMtuUseXs() const {
            return SmallMtuUseXs;
        }
        void SetUseTosCongestionAlgo(const bool useTosCongestionAlgo) {
            UseTosCongestionAlgo = useTosCongestionAlgo;
        }
        void SetInflateCongestion(const bool inflateCongestion) {
            InflateCongestion = inflateCongestion;
        }
        void SetSmallMtuUseXs(const bool smallMtuUseXs) {
            SmallMtuUseXs = smallMtuUseXs;
        }

    private:
        bool UseTosCongestionAlgo;
        bool InflateCongestion;
        bool SmallMtuUseXs;
    };

    inline bool operator==(const TConnectionSettings& lhv, const TConnectionSettings& rhv) {
        return lhv.GetUseTosCongestionAlgo() == rhv.GetUseTosCongestionAlgo() &&
               lhv.GetInflateCongestion() == rhv.GetInflateCongestion() &&
               lhv.GetSmallMtuUseXs() == rhv.GetSmallMtuUseXs();
    }
    inline bool operator!=(const TConnectionSettings& lhv, const TConnectionSettings& rhv) {
        return !(lhv == rhv);
    }

    inline size_t TConnectionSettingsHash(const TConnectionSettings& v) {
        return (size_t)v.GetUseTosCongestionAlgo() + ((size_t)v.GetInflateCongestion() << 1);
    }
}

template <>
struct THash<NNetliba_v12::TConnectionSettings> {
    inline size_t operator()(const NNetliba_v12::TConnectionSettings& settings) const {
        return NNetliba_v12::TConnectionSettingsHash(settings);
    }
};
