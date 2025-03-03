/* -----------------------------------------------------------------------------
 * std_deque.i
 *
 * Default std_deque wrapper
 * ----------------------------------------------------------------------------- */

%module std_deque

%rename(__getitem__) std::deque::getitem;
%rename(__setitem__) std::deque::setitem;
%rename(__delitem__) std::deque::delitem;
%rename(__getslice__) std::deque::getslice;
%rename(__setslice__) std::deque::setslice;
%rename(__delslice__) std::deque::delslice;

%extend std::deque {
   int __len__() {
       return (int) self->size();
   }
   int __nonzero__() {
       return ! self->empty();
   }
   void append(const T &x) {
       self->push_back(x);
   }
};

%include <std/_std_deque.i>
