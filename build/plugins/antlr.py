ANTLR_RESOURCE_ID = '164589140'
ANTLR_JAR_PATH = 'antlr-3.5.2-complete-no-st3.jar'


def onrun_antlr(unit, *args):
    unit.onexternal_resource(['ANTLR', 'sbr:' + ANTLR_RESOURCE_ID])

    # XXX workaround ymake swag behavior
    unit.set(['ANTLR', '$(ANTLR)'])
    unit.onrun_java(['-jar', '${ANTLR}/' + ANTLR_JAR_PATH] + list(args))
