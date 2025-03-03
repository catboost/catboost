/* -----------------------------------------------------------------------------
 * Ruby API portion that goes into the runtime
 * ----------------------------------------------------------------------------- */

#ifdef __cplusplus
extern "C" {
#endif

SWIGINTERN VALUE
SWIG_Ruby_AppendOutput(VALUE target, VALUE o) {
  if (NIL_P(target)) {
    target = o;
  } else {
    if (TYPE(target) != T_ARRAY) {
      VALUE o2 = target;
      target = rb_ary_new();
      rb_ary_push(target, o2);
    }
    rb_ary_push(target, o);
  }
  return target;
}

/* For ruby1.8.4 and earlier. */
#ifndef RUBY_INIT_STACK
   RUBY_EXTERN void Init_stack(VALUE* addr);
#  define RUBY_INIT_STACK \
   VALUE variable_in_this_stack_frame; \
   Init_stack(&variable_in_this_stack_frame);
#endif


#ifdef __cplusplus
}
#endif

