/* -----------------------------------------------------------------------------
 * rubyautodoc.swg
 *
 * This file implements autodoc typemaps for some common ruby methods.
 * ----------------------------------------------------------------------------- */

%define AUTODOC(func, str)
  %feature("autodoc", str) func;
%enddef


AUTODOC(to_i, "Convert $class to an Integer");
AUTODOC(to_f, "Convert $class to a Float");
AUTODOC(coerce, "Coerce class to a number");
AUTODOC(to_a, "Convert $class to an Array");
AUTODOC(to_s, "Convert class to a String representation");
AUTODOC(inspect, "Inspect class and its contents");

AUTODOC(at, "Return element at a certain index");
AUTODOC(__getitem__, "Element accessor/slicing");
AUTODOC(__setitem__, "Element setter/slicing");
AUTODOC(slice, "Return a slice (portion of) the $class");

AUTODOC(push, "Add an element at the end of the $class");
AUTODOC(pop, "Remove and return element at the end of the $class");
AUTODOC(shift, "Remove and return element at the beginning of the $class");
AUTODOC(unshift, "Add one or more elements at the beginning of the $class");
AUTODOC(first, "Return the first element in $class");
AUTODOC(last, "Return the last element in $class");


//
// Common Object methods
//
AUTODOC(hash, "Hashing function for class");
AUTODOC(dup, "Create a duplicate of the class and unfreeze it if needed");
AUTODOC(clone, "Create a duplicate of the class");

//
// Container methods
//
AUTODOC(empty, "Check if $class is empty");
AUTODOC(size, "Size or Length of the $class");
AUTODOC(insert, "Insert one or more new elements in the $class");

//
// Iterator methods (block)
//
AUTODOC(each, "Iterate thru each element in the $class.  A block must be provided");
AUTODOC(find, "Find an element in the class");
AUTODOC(each_key, "Iterate thru each key element in the $class.  A block must be provided");
AUTODOC(each_value, "Iterate thru each key element in the $class.  A block must be provided");
AUTODOC(reject, "Iterate thru each element in the $class and reject those that fail a condition returning a new $class.  A block must be provided");
AUTODOC(reject_bang, "Iterate thru each element in the $class and reject those that fail a condition.  A block must be provided.  $class is modified in place");
AUTODOC(select, "Iterate thru each element in the $class and select those that match a condition.  A block must be provided");
AUTODOC(delete_at, "Delete an element at a certain index");
AUTODOC(__delete__, "Delete a matching element");


//
// Hash methods
//
AUTODOC(keys, "Return an Array of key elements");
AUTODOC(values, "Return an Array of value elements");
AUTODOC(values_at, "Return an Array of value elements matching the conditions");


//
// Operators
//
#ifdef __cplusplus
AUTODOC(operator==, "Equality comparison operator");
AUTODOC(operator<=, "Lower or equal comparison operator");
AUTODOC(operator>=, "Higher or equal comparison operator");
AUTODOC(operator<, "Lower than comparison operator");
AUTODOC(operator>, "Higher than comparison operator");
AUTODOC(operator<<, "Left shifting or appending operator");
AUTODOC(operator>>, "Right shifting operator or extracting operator");
AUTODOC(operator+, "Add operator");
AUTODOC(operator-, "Substraction operator");
AUTODOC(operator+(), "Positive operator");
AUTODOC(operator-(), "Negation operator");
AUTODOC(operator&, "AND operator");
AUTODOC(operator|, "OR operator");
AUTODOC(operator^, "XOR operator");
AUTODOC(operator~, "Invert operator");
#endif
AUTODOC(__eq__, "Equality comparison operator");
AUTODOC(__le__, "Lower or equal comparison operator");
AUTODOC(__ge__, "Higher or equal comparison operator");
AUTODOC(__lt__, "Lower than comparison operator");
AUTODOC(__gt__, "Higher than comparison operator");
AUTODOC(__lshift__, "Left shifting or appending operator");
AUTODOC(__rshift__, "Right shifting operator or extracting operator");
AUTODOC(__add___, "Add operator");
AUTODOC(__sub__, "Substraction operator");
AUTODOC(__pos__, "Positive operator");
AUTODOC(__neg__, "Negation operator");
AUTODOC(__and__, "AND operator");
AUTODOC(__or__, "OR operator");
AUTODOC(__xor__, "XOR operator");
AUTODOC(__negate__, "Invert operator");
AUTODOC(__pow__, "Exponential operator");
AUTODOC(__divmod__, "Modulo of division");
AUTODOC(__cmp__, "Comparison operator.  Returns < 0 for less than, 0 for equal or > 1 for higher than.");
