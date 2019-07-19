LIBRARY()



#TODO(kirillovs): invert include logic
# better replace thin with model/fat wich will include all catboost model possibilities
PEERDIR(
    catboost/libs/model/thin
    catboost/libs/model/model_export
)

END()
