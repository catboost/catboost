#include "tensorboard_logger.h"

uint32_t Mask(uint32_t crc) {
    return ((crc >> 15) | (crc << 17)) + 0xa282ead8ul;
}

int TTensorBoardLogger::Write(tensorboard::Event& event) {
    TString buf;
    Y_PROTOBUF_SUPPRESS_NODISCARD event.SerializeToString(&buf);
    uint64_t bufLen = static_cast<uint64_t>(buf.size());
    uint32_t lenCrc = Mask(Crc32c((char*)&bufLen, sizeof(uint64_t)));
    uint32_t dataCrc = Mask(Crc32c(buf.c_str(), buf.size()));

    OutputStream->Write((char*)&bufLen, sizeof(uint64_t));
    OutputStream->Write((char*)&lenCrc, sizeof(uint32_t));
    OutputStream->Write(buf.c_str(), buf.size());
    OutputStream->Write((char*)&dataCrc, sizeof(uint32_t));
    OutputStream->Flush();
    return 0;
}

int TTensorBoardLogger::AddEvent(int64_t step, THolder<tensorboard::Summary>* summary) {
    tensorboard::Event event;
    double wallTime = time(NULL);

    event.set_wall_time(wallTime);
    event.set_step(step);
    event.set_allocated_summary(summary->Release());
    Write(event);
    return 0;
}

TTensorBoardLogger::TTensorBoardLogger(const TString& logDir) {
    if (!logDir.empty()) {
        MakePathIfNotExist(logDir.c_str());
    }
    TString logFile = JoinFsPaths(logDir, "events.out.tfevents");
    OutputStream = THolder<TOFStream>(new TOFStream(logFile));
}

int TTensorBoardLogger::AddScalar(const TString& tag, int step, float value) {
    THolder<tensorboard::Summary> summary(new tensorboard::Summary());
    auto summaryValue = summary->add_value();

    summaryValue->set_node_name(tag);
    summaryValue->set_tag(tag);
    summaryValue->set_simple_value(value);
    AddEvent(step, &summary);
    return 0;
}

