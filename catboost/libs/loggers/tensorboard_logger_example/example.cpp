#include "catboost/libs/loggers/tensorboard_logger.h"

int main() {
    TTensorBoardLogger logger("./first_experiment");

    logger.AddScalar("acc", 0, 0.1);
    logger.AddScalar("loss", 0, 1.5);
    logger.AddScalar("acc", 1, 0.5);
    logger.AddScalar("loss", 1, 0.8);
    logger.AddScalar("acc", 2, 0.75);
    logger.AddScalar("loss", 2, 0.5);
    logger.AddScalar("acc", 3, 0.8);
    logger.AddScalar("loss", 3, 0.4);

    TTensorBoardLogger logger2("./second_experiment");

    logger2.AddScalar("acc", 0, 1.1);
    logger2.AddScalar("loss", 0, 2.5);
    logger2.AddScalar("acc", 1, 1.5);
    logger2.AddScalar("loss", 1, 1.8);
    logger2.AddScalar("acc", 2, 1.75);
    logger2.AddScalar("loss", 2, 1.5);
    logger2.AddScalar("acc", 3, 1.8);
    logger2.AddScalar("loss", 3, 1.4);

    return 0;
}
