;;;************************************************************************
;;;*common.scm
;;;*
;;;*     This file contains generic SWIG GOOPS classes for generated
;;;*     GOOPS file support
;;;************************************************************************

(define-module (Swig swigrun))

(define-module (Swig common)
  #:use-module (oop goops)
  #:use-module (Swig swigrun))

(define-class <swig-metaclass> (<class>)
  (new-function #:init-value #f))

(define-method (initialize (class <swig-metaclass>) initargs)
  (slot-set! class 'new-function (get-keyword #:new-function initargs #f))
  (next-method))

(define-class <swig> () 
  (swig-smob #:init-value #f)
  #:metaclass <swig-metaclass>
)

(define-method (initialize (obj <swig>) initargs)
  (next-method)
  (slot-set! obj 'swig-smob
    (let ((arg (get-keyword #:init-smob initargs #f)))
      (if arg
        arg
        (let ((ret (apply (slot-ref (class-of obj) 'new-function) (get-keyword #:args initargs '()))))
          ;; if the class is registered with runtime environment,
          ;; new-Function will return a <swig> goops class.  In that case, extract the smob
          ;; from that goops class and set it as the current smob.
          (if (slot-exists? ret 'swig-smob)
            (slot-ref ret 'swig-smob)
            ret))))))

(define (display-address o file)
  (display (number->string (object-address o) 16) file))

(define (display-pointer-address o file)
  ;; Don't fail if the function SWIG-PointerAddress is not present.
  (let ((address (false-if-exception (SWIG-PointerAddress o))))
    (if address
	(begin
	  (display " @ " file)
	  (display (number->string address 16) file)))))

(define-method (write (o <swig>) file)
  ;; We display _two_ addresses to show the object's identity:
  ;;  * first the address of the GOOPS proxy object,
  ;;  * second the pointer address.
  ;; The reason is that proxy objects are created and discarded on the
  ;; fly, so different proxy objects for the same C object will appear.
  (let ((class (class-of o)))
    (if (slot-bound? class 'name)
	(begin
	  (display "#<" file)
	  (display (class-name class) file)
	  (display #\space file)
	  (display-address o file)
	  (display-pointer-address o file)
	  (display ">" file))
	(next-method))))
                                              
(export <swig-metaclass> <swig>)

;;; common.scm ends here
