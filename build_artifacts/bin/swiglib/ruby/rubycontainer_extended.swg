/* -----------------------------------------------------------------------------
 * rubycontainer_extended.swg
 *
 * This file contains additional functions that make containers
 * behave closer to ruby primitive types.
 * However, some of these functions place some restrictions on
 * the underlying object inside of the container and the iterator
 * (that it has to have an == comparison function, that it has to have 
 * an = assignment operator, etc).
 * ----------------------------------------------------------------------------- */

/** 
 * Macro used to add extend functions that require operator== in object.
 * 
 * @param Container    STL container
 * @param Type         class inside container
 *
 */
%define %swig_container_with_equal_operator( Container, Type )

  VALUE __delete__( const Type& val ) {
    VALUE r = Qnil;
    Container<Type >::iterator e = $self->end();
    Container<Type >::iterator i = std::remove( $self->begin(), e, val );
    // remove dangling elements now
    $self->erase( i, e );
    
    if ( i != e )
      r = swig::from< Type >( val );
    else if ( rb_block_given_p() )
      r = rb_yield(Qnil);
    return r;
  }

%enddef  // end of  %swig_container_with_equal_operator




/** 
 * Macro used to add extend functions that require the assignment 
 * operator (ie. = ) of contained class
 * 
 * @param Container    STL container
 * @param Type         class inside container
 * 
 */

%define %swig_container_with_assignment( Container, Type )


  //
  // map!  -- the equivalent of std::transform
  //
  Container< Type >* map_bang() {

    if ( !rb_block_given_p() )
      rb_raise( rb_eArgError, "No block given" );

    VALUE r = Qnil;
    Container< Type >::iterator i = $self->begin();
    Container< Type >::iterator e = $self->end();

    try {
      for ( ; i != e; ++i )
	{
	  r = swig::from< Type >( *i );
	  r = rb_yield( r );
	  *i = swig::as< Type >( r );
	}
    }
    catch (const std::invalid_argument&)
      {
	rb_raise(rb_eTypeError,
		 "Yield block did not return a valid element for " "Container");
      }
    
    return $self;
  }


%enddef  // end of  %swig_container_with_assignment





/** 
 * Macro used to add all extended functions to a container
 * 
 * @param Container    STL container
 * @param Type         class inside container
 * 
 */
%define %swig_container_extend( Container, Type )

%extend Container< Type > {

  %swig_container_with_assignment( %arg(Container), Type );
  %swig_container_with_equal_operator( %arg(Container), Type );

}

%enddef


/** 
 * Private macro used to add all extended functions to C/C++
 * primitive types
 * 
 * @param Container an STL container, like std::vector (with no class template)
 *
 */
%define %__swig_container_extend_primtypes( Container )

%swig_container_extend( %arg( Container ), bool );
%swig_container_extend( %arg( Container ), char );
%swig_container_extend( %arg( Container ), short );
%swig_container_extend( %arg( Container ), int );
%swig_container_extend( %arg( Container ), unsigned short );
%swig_container_extend( %arg( Container ), unsigned int );
%swig_container_extend( %arg( Container ), float );
%swig_container_extend( %arg( Container ), double );
%swig_container_extend( %arg( Container ), std::complex );
%swig_container_extend( %arg( Container ), std::string );
%swig_container_extend( %arg( Container ), swig::GC_VALUE );

%enddef


%__swig_container_extend_primtypes( std::vector );
%__swig_container_extend_primtypes( std::deque );
%__swig_container_extend_primtypes( std::list );

