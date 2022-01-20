#pragma once

////////////////////////////////////////////////////////////////////////////////

//! Declares a trivial public read-write property that is passed by reference.
#define DECLARE_BYREF_RW_PROPERTY(type, name) \
public: \
    type& name(); \
    const type& name() const;

//! Defines a trivial public read-write property that is passed by reference.
//! All arguments after name are used as default value (via braced-init-list).
#define DEFINE_BYREF_RW_PROPERTY(type, name, ...) \
protected: \
    type name##_ { __VA_ARGS__ }; \
    \
public: \
    Y_FORCE_INLINE type& name() \
    { \
        return name##_; \
    } \
    \
    Y_FORCE_INLINE const type& name() const \
    { \
        return name##_; \
    }

//! Defines a trivial public read-write property that is passed by reference
//! and is not inline-initialized.
#define DEFINE_BYREF_RW_PROPERTY_NO_INIT(type, name) \
protected: \
    type name##_; \
    \
public: \
    Y_FORCE_INLINE type& name() \
    { \
        return name##_; \
    } \
    \
    Y_FORCE_INLINE const type& name() const \
    { \
        return name##_; \
    }

//! Forwards a trivial public read-write property that is passed by reference.
#define DELEGATE_BYREF_RW_PROPERTY(declaringType, type, name, delegateTo) \
    type& declaringType::name() \
    { \
        return (delegateTo).name(); \
    } \
    \
    const type& declaringType::name() const \
    { \
        return (delegateTo).name(); \
    }

////////////////////////////////////////////////////////////////////////////////

//! Declares a trivial public read-only property that is passed by reference.
#define DECLARE_BYREF_RO_PROPERTY(type, name) \
public: \
    const type& name() const;

//! Defines a trivial public read-only property that is passed by reference.
//! All arguments after name are used as default value (via braced-init-list).
#define DEFINE_BYREF_RO_PROPERTY(type, name, ...) \
protected: \
    type name##_ { __VA_ARGS__ }; \
    \
public: \
    Y_FORCE_INLINE const type& name() const \
    { \
        return name##_; \
    }

//! Defines a trivial public read-only property that is passed by reference
//! and is not inline-initialized.
#define DEFINE_BYREF_RO_PROPERTY_NO_INIT(type, name) \
protected: \
    type name##_; \
    \
public: \
    Y_FORCE_INLINE const type& name() const \
    { \
        return name##_; \
    }

//! Forwards a trivial public read-only property that is passed by reference.
#define DELEGATE_BYREF_RO_PROPERTY(declaringType, type, name, delegateTo) \
    const type& declaringType::name() const \
    { \
        return (delegateTo).name(); \
    }

////////////////////////////////////////////////////////////////////////////////

//! Declares a trivial public read-write property that is passed by value.
#define DECLARE_BYVAL_RW_PROPERTY(type, name) \
public: \
    type Get##name() const; \
    void Set##name(type value);

//! Defines a trivial public read-write property that is passed by value.
//! All arguments after name are used as default value (via braced-init-list).
#define DEFINE_BYVAL_RW_PROPERTY(type, name, ...) \
protected: \
    type name##_ { __VA_ARGS__ }; \
    \
public: \
    Y_FORCE_INLINE type Get##name() const \
    { \
        return name##_; \
    } \
    \
    Y_FORCE_INLINE void Set##name(type value) \
    { \
        name##_ = value; \
    } \

//! Defines a trivial public read-write property that is passed by value.
//! All arguments after name are used as default value (via braced-init-list).
#define DEFINE_BYVAL_RW_PROPERTY_WITH_FLUENT_SETTER(declaringType, type, name, ...) \
protected: \
    type name##_ { __VA_ARGS__ }; \
    \
public: \
    Y_FORCE_INLINE type Get##name() const \
    { \
        return name##_; \
    } \
    \
    Y_FORCE_INLINE void Set##name(type value) &\
    { \
        name##_ = value; \
    } \
    \
    Y_FORCE_INLINE declaringType&& Set##name(type value) &&\
    { \
        name##_ = value; \
        return std::move(*this); \
    } \

//! Defines a trivial public read-write property that is passed by value
//! and is not inline-initialized.
#define DEFINE_BYVAL_RW_PROPERTY_NO_INIT(type, name, ...) \
protected: \
    type name##_; \
    \
public: \
    Y_FORCE_INLINE type Get##name() const \
    { \
        return name##_; \
    } \
    \
    Y_FORCE_INLINE void Set##name(type value) \
    { \
        name##_ = value; \
    } \

//! Forwards a trivial public read-write property that is passed by value.
#define DELEGATE_BYVAL_RW_PROPERTY(declaringType, type, name, delegateTo) \
    type declaringType::Get##name() const \
    { \
        return (delegateTo).Get##name(); \
    } \
    \
    void declaringType::Set##name(type value) \
    { \
        (delegateTo).Set##name(value); \
    }

////////////////////////////////////////////////////////////////////////////////

//! Declares a trivial public read-only property that is passed by value.
#define DECLARE_BYVAL_RO_PROPERTY(type, name) \
public: \
    type Get##name() const;

//! Defines a trivial public read-only property that is passed by value.
//! All arguments after name are used as default value (via braced-init-list).
#define DEFINE_BYVAL_RO_PROPERTY(type, name, ...) \
protected: \
    type name##_ { __VA_ARGS__ }; \
    \
public: \
    Y_FORCE_INLINE type Get##name() const \
    { \
        return name##_; \
    }


//! Defines a trivial public read-only property that is passed by value
//! and is not inline-initialized.
#define DEFINE_BYVAL_RO_PROPERTY_NO_INIT(type, name) \
protected: \
    type name##_; \
    \
public: \
    Y_FORCE_INLINE type Get##name() const \
    { \
        return name##_; \
    }

//! Forwards a trivial public read-only property that is passed by value.
#define DELEGATE_BYVAL_RO_PROPERTY(declaringType, type, name, delegateTo) \
    type declaringType::Get##name() \
    { \
        return (delegateTo).Get##name(); \
    }

////////////////////////////////////////////////////////////////////////////////

//! Below are macro helpers for extra properties.
//! Extra properties should be used for lazy memory allocation for properties that
//! hold default values for the majority of objects.

//! Initializes extra property holder if it is not initialized.
#define INITIALIZE_EXTRA_PROPERTY_HOLDER(holder) \
    if (!holder##_) { \
        holder##_.reset(new decltype(holder##_)::element_type()); \
    }

//! Declares an extra property holder. Holder contains extra properties values.
//! Holder is not created until some property is set with a non-default value.
//! If there is no holder property getter returns default value.
#define DECLARE_EXTRA_PROPERTY_HOLDER(type, holder) \
public: \
    Y_FORCE_INLINE bool HasCustom##holder() const \
    { \
        return static_cast<bool>(holder##_); \
    } \
    Y_FORCE_INLINE const type* GetCustom##holder() const \
    { \
        return holder##_.get(); \
    } \
    Y_FORCE_INLINE type* GetCustom##holder() \
    { \
        return holder##_.get(); \
    } \
    Y_FORCE_INLINE void InitializeCustom##holder() \
    { \
        INITIALIZE_EXTRA_PROPERTY_HOLDER(holder) \
    } \
private: \
    std::unique_ptr<type> holder##_; \
    static const type Default##holder##_;

//! Defines a storage for extra properties default values.
#define DEFINE_EXTRA_PROPERTY_HOLDER(class, type, holder) \
    const type class::Default##holder##_;

//! Defines a public read-write extra property that is passed by value.
#define DEFINE_BYVAL_RW_EXTRA_PROPERTY(holder, name) \
public: \
    Y_FORCE_INLINE decltype(holder##_->name) Get##name() const \
    { \
        if (!holder##_) { \
            return Default##holder##_.name; \
        } \
        return holder##_->name; \
    } \
    Y_FORCE_INLINE void Set##name(decltype(holder##_->name) val) \
    { \
        if (!holder##_) { \
            if (val == Default##holder##_.name) { \
                return; \
            } \
            INITIALIZE_EXTRA_PROPERTY_HOLDER(holder); \
        } \
        holder##_->name = val; \
    }

//! Defines a public read-write extra property that is passed by reference.
#define DEFINE_BYREF_RW_EXTRA_PROPERTY(holder, name) \
public: \
    Y_FORCE_INLINE const decltype(holder##_->name)& name() const \
    { \
        if (!holder##_) { \
            return Default##holder##_.name; \
        } \
        return holder##_->name; \
    } \
    Y_FORCE_INLINE decltype(holder##_->name)& Mutable##name() \
    { \
        INITIALIZE_EXTRA_PROPERTY_HOLDER(holder); \
        return holder##_->name; \
    }

////////////////////////////////////////////////////////////////////////////////
