ANTLR_JAR_PATH = 'antlr-3.5.2-complete-no-st3.jar'


def onrun_antlr(unit, *args):
    unit.onpeerdir(['build/external_resources/antlr3'])
    unit.on_run_java(['-jar', '$ANTLR3_RESOURCE_GLOBAL/' + ANTLR_JAR_PATH] + list(args))
