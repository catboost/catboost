/* -----------------------------------------------------------------------------
 * interpreter.i
 *
 * SWIG file for a simple Guile interpreter
 * ----------------------------------------------------------------------------- */

%{

#include <stdio.h>
GSCM_status guile_init();

int main(int argc, char **argv) {
  GSCM_status status;
  GSCM_top_level toplev;
  char *eval_answer;
  char input_str[16384];
  int done;

  /* start a scheme interpreter */
  status = gscm_run_scm(argc, argv, 0, stdout, stderr, guile_init, 0, "#t");
  if (status != GSCM_OK) {
    fputs(gscm_error_msg(status), stderr);
    fputc('\n', stderr);
    printf("Error in startup.\n");
    exit(1);
  }

  /* create the top level environment */
  status = gscm_create_top_level(&toplev);
  if (status != GSCM_OK) {
    fputs(gscm_error_msg(status), stderr);
    fputc('\n', stderr);
    exit(1);
  }

  /* now sit in a scheme eval loop: I input the expressions, have guile
   * evaluate them, and then get another expression.
   */
  done = 0;
  fprintf(stdout,"Guile > ");
  while (!done) {
    if (fgets(input_str,16384,stdin) == NULL) {
      exit(1);
    } else {
      if (strncmp(input_str,"quit",4) == 0) exit(1);
      status = gscm_eval_str(&eval_answer, toplev, input_str);
      fprintf(stdout,"%s\n", eval_answer);
      fprintf(stdout,"Guile > ");
    }
  }

  /* now clean up and quit */
  gscm_destroy_top_level(toplev);
}

%}



