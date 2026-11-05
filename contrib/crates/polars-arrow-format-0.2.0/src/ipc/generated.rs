pub use root::*;

const _: () = ::planus::check_version_compatibility("planus-1.1.1");

/// The root namespace
///
/// Generated from these locations:
/// * File `File.fbs`
#[no_implicit_prelude]
#[allow(dead_code, clippy::needless_lifetimes)]
mod root {
    /// The namespace `org`
    ///
    /// Generated from these locations:
    /// * File `File.fbs`
    pub mod org {
        /// The namespace `org.apache`
        ///
        /// Generated from these locations:
        /// * File `File.fbs`
        pub mod apache {
            /// The namespace `org.apache.arrow`
            ///
            /// Generated from these locations:
            /// * File `File.fbs`
            pub mod arrow {
                /// The namespace `org.apache.arrow.flatbuf`
                ///
                /// Generated from these locations:
                /// * File `File.fbs`
                /// * File `Schema.fbs`
                /// * File `Message.fbs`
                /// * File `SparseTensor.fbs`
                /// * File `Tensor.fbs`
                pub mod flatbuf {
                    ///  ----------------------------------------------------------------------
                    ///  Arrow File metadata
                    ///
                    ///
                    /// Generated from these locations:
                    /// * Table `Footer` in the file `File.fbs:26`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct Footer {
                        /// The field `version` in the table `Footer`
                        pub version: self::MetadataVersion,
                        /// The field `schema` in the table `Footer`
                        pub schema:
                            ::core::option::Option<::planus::alloc::boxed::Box<self::Schema>>,
                        /// The field `dictionaries` in the table `Footer`
                        pub dictionaries:
                            ::core::option::Option<::planus::alloc::vec::Vec<self::Block>>,
                        /// The field `recordBatches` in the table `Footer`
                        pub record_batches:
                            ::core::option::Option<::planus::alloc::vec::Vec<self::Block>>,
                        ///  User-defined metadata
                        pub custom_metadata:
                            ::core::option::Option<::planus::alloc::vec::Vec<self::KeyValue>>,
                    }

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for Footer {
                        fn default() -> Self {
                            Self {
                                version: self::MetadataVersion::V1,
                                schema: ::core::default::Default::default(),
                                dictionaries: ::core::default::Default::default(),
                                record_batches: ::core::default::Default::default(),
                                custom_metadata: ::core::default::Default::default(),
                            }
                        }
                    }

                    impl Footer {
                        /// Creates a [FooterBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> FooterBuilder<()> {
                            FooterBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            field_version: impl ::planus::WriteAsDefault<
                                self::MetadataVersion,
                                self::MetadataVersion,
                            >,
                            field_schema: impl ::planus::WriteAsOptional<::planus::Offset<self::Schema>>,
                            field_dictionaries: impl ::planus::WriteAsOptional<
                                ::planus::Offset<[self::Block]>,
                            >,
                            field_record_batches: impl ::planus::WriteAsOptional<
                                ::planus::Offset<[self::Block]>,
                            >,
                            field_custom_metadata: impl ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::KeyValue>]>,
                            >,
                        ) -> ::planus::Offset<Self> {
                            let prepared_version =
                                field_version.prepare(builder, &self::MetadataVersion::V1);
                            let prepared_schema = field_schema.prepare(builder);
                            let prepared_dictionaries = field_dictionaries.prepare(builder);
                            let prepared_record_batches = field_record_batches.prepare(builder);
                            let prepared_custom_metadata = field_custom_metadata.prepare(builder);

                            let mut table_writer: ::planus::table_writer::TableWriter<14> =
                                ::core::default::Default::default();
                            if prepared_schema.is_some() {
                                table_writer.write_entry::<::planus::Offset<self::Schema>>(1);
                            }
                            if prepared_dictionaries.is_some() {
                                table_writer.write_entry::<::planus::Offset<[self::Block]>>(2);
                            }
                            if prepared_record_batches.is_some() {
                                table_writer.write_entry::<::planus::Offset<[self::Block]>>(3);
                            }
                            if prepared_custom_metadata.is_some() {
                                table_writer.write_entry::<::planus::Offset<[::planus::Offset<self::KeyValue>]>>(4);
                            }
                            if prepared_version.is_some() {
                                table_writer.write_entry::<self::MetadataVersion>(0);
                            }

                            unsafe {
                                table_writer.finish(builder, |object_writer| {
                                    if let ::core::option::Option::Some(prepared_schema) =
                                        prepared_schema
                                    {
                                        object_writer.write::<_, _, 4>(&prepared_schema);
                                    }
                                    if let ::core::option::Option::Some(prepared_dictionaries) =
                                        prepared_dictionaries
                                    {
                                        object_writer.write::<_, _, 4>(&prepared_dictionaries);
                                    }
                                    if let ::core::option::Option::Some(prepared_record_batches) =
                                        prepared_record_batches
                                    {
                                        object_writer.write::<_, _, 4>(&prepared_record_batches);
                                    }
                                    if let ::core::option::Option::Some(prepared_custom_metadata) =
                                        prepared_custom_metadata
                                    {
                                        object_writer.write::<_, _, 4>(&prepared_custom_metadata);
                                    }
                                    if let ::core::option::Option::Some(prepared_version) =
                                        prepared_version
                                    {
                                        object_writer.write::<_, _, 2>(&prepared_version);
                                    }
                                });
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Footer>> for Footer {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Footer> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Footer>> for Footer {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Footer>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Footer> for Footer {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Footer> {
                            Footer::create(
                                builder,
                                self.version,
                                &self.schema,
                                &self.dictionaries,
                                &self.record_batches,
                                &self.custom_metadata,
                            )
                        }
                    }

                    /// Builder for serializing an instance of the [Footer] type.
                    ///
                    /// Can be created using the [Footer::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct FooterBuilder<State>(State);

                    impl FooterBuilder<()> {
                        /// Setter for the [`version` field](Footer#structfield.version).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn version<T0>(self, value: T0) -> FooterBuilder<(T0,)>
                        where
                            T0: ::planus::WriteAsDefault<
                                self::MetadataVersion,
                                self::MetadataVersion,
                            >,
                        {
                            FooterBuilder((value,))
                        }

                        /// Sets the [`version` field](Footer#structfield.version) to the default value.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn version_as_default(
                            self,
                        ) -> FooterBuilder<(::planus::DefaultValue,)> {
                            self.version(::planus::DefaultValue)
                        }
                    }

                    impl<T0> FooterBuilder<(T0,)> {
                        /// Setter for the [`schema` field](Footer#structfield.schema).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn schema<T1>(self, value: T1) -> FooterBuilder<(T0, T1)>
                        where
                            T1: ::planus::WriteAsOptional<::planus::Offset<self::Schema>>,
                        {
                            let (v0,) = self.0;
                            FooterBuilder((v0, value))
                        }

                        /// Sets the [`schema` field](Footer#structfield.schema) to null.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn schema_as_null(self) -> FooterBuilder<(T0, ())> {
                            self.schema(())
                        }
                    }

                    impl<T0, T1> FooterBuilder<(T0, T1)> {
                        /// Setter for the [`dictionaries` field](Footer#structfield.dictionaries).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn dictionaries<T2>(self, value: T2) -> FooterBuilder<(T0, T1, T2)>
                        where
                            T2: ::planus::WriteAsOptional<::planus::Offset<[self::Block]>>,
                        {
                            let (v0, v1) = self.0;
                            FooterBuilder((v0, v1, value))
                        }

                        /// Sets the [`dictionaries` field](Footer#structfield.dictionaries) to null.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn dictionaries_as_null(self) -> FooterBuilder<(T0, T1, ())> {
                            self.dictionaries(())
                        }
                    }

                    impl<T0, T1, T2> FooterBuilder<(T0, T1, T2)> {
                        /// Setter for the [`recordBatches` field](Footer#structfield.record_batches).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn record_batches<T3>(
                            self,
                            value: T3,
                        ) -> FooterBuilder<(T0, T1, T2, T3)>
                        where
                            T3: ::planus::WriteAsOptional<::planus::Offset<[self::Block]>>,
                        {
                            let (v0, v1, v2) = self.0;
                            FooterBuilder((v0, v1, v2, value))
                        }

                        /// Sets the [`recordBatches` field](Footer#structfield.record_batches) to null.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn record_batches_as_null(self) -> FooterBuilder<(T0, T1, T2, ())> {
                            self.record_batches(())
                        }
                    }

                    impl<T0, T1, T2, T3> FooterBuilder<(T0, T1, T2, T3)> {
                        /// Setter for the [`custom_metadata` field](Footer#structfield.custom_metadata).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn custom_metadata<T4>(
                            self,
                            value: T4,
                        ) -> FooterBuilder<(T0, T1, T2, T3, T4)>
                        where
                            T4: ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::KeyValue>]>,
                            >,
                        {
                            let (v0, v1, v2, v3) = self.0;
                            FooterBuilder((v0, v1, v2, v3, value))
                        }

                        /// Sets the [`custom_metadata` field](Footer#structfield.custom_metadata) to null.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn custom_metadata_as_null(
                            self,
                        ) -> FooterBuilder<(T0, T1, T2, T3, ())> {
                            self.custom_metadata(())
                        }
                    }

                    impl<T0, T1, T2, T3, T4> FooterBuilder<(T0, T1, T2, T3, T4)> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Footer].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Footer>
                        where
                            Self: ::planus::WriteAsOffset<Footer>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<self::MetadataVersion, self::MetadataVersion>,
                            T1: ::planus::WriteAsOptional<::planus::Offset<self::Schema>>,
                            T2: ::planus::WriteAsOptional<::planus::Offset<[self::Block]>>,
                            T3: ::planus::WriteAsOptional<::planus::Offset<[self::Block]>>,
                            T4: ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::KeyValue>]>,
                            >,
                        > ::planus::WriteAs<::planus::Offset<Footer>>
                        for FooterBuilder<(T0, T1, T2, T3, T4)>
                    {
                        type Prepared = ::planus::Offset<Footer>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Footer> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<self::MetadataVersion, self::MetadataVersion>,
                            T1: ::planus::WriteAsOptional<::planus::Offset<self::Schema>>,
                            T2: ::planus::WriteAsOptional<::planus::Offset<[self::Block]>>,
                            T3: ::planus::WriteAsOptional<::planus::Offset<[self::Block]>>,
                            T4: ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::KeyValue>]>,
                            >,
                        > ::planus::WriteAsOptional<::planus::Offset<Footer>>
                        for FooterBuilder<(T0, T1, T2, T3, T4)>
                    {
                        type Prepared = ::planus::Offset<Footer>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Footer>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<self::MetadataVersion, self::MetadataVersion>,
                            T1: ::planus::WriteAsOptional<::planus::Offset<self::Schema>>,
                            T2: ::planus::WriteAsOptional<::planus::Offset<[self::Block]>>,
                            T3: ::planus::WriteAsOptional<::planus::Offset<[self::Block]>>,
                            T4: ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::KeyValue>]>,
                            >,
                        > ::planus::WriteAsOffset<Footer> for FooterBuilder<(T0, T1, T2, T3, T4)>
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Footer> {
                            let (v0, v1, v2, v3, v4) = &self.0;
                            Footer::create(builder, v0, v1, v2, v3, v4)
                        }
                    }

                    /// Reference to a deserialized [Footer].
                    #[derive(Copy, Clone)]
                    pub struct FooterRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> FooterRef<'a> {
                        /// Getter for the [`version` field](Footer#structfield.version).
                        #[inline]
                        pub fn version(&self) -> ::planus::Result<self::MetadataVersion> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(0, "Footer", "version")?
                                    .unwrap_or(self::MetadataVersion::V1),
                            )
                        }

                        /// Getter for the [`schema` field](Footer#structfield.schema).
                        #[inline]
                        pub fn schema(
                            &self,
                        ) -> ::planus::Result<::core::option::Option<self::SchemaRef<'a>>>
                        {
                            self.0.access(1, "Footer", "schema")
                        }

                        /// Getter for the [`dictionaries` field](Footer#structfield.dictionaries).
                        #[inline]
                        pub fn dictionaries(
                            &self,
                        ) -> ::planus::Result<
                            ::core::option::Option<::planus::Vector<'a, self::BlockRef<'a>>>,
                        > {
                            self.0.access(2, "Footer", "dictionaries")
                        }

                        /// Getter for the [`recordBatches` field](Footer#structfield.record_batches).
                        #[inline]
                        pub fn record_batches(
                            &self,
                        ) -> ::planus::Result<
                            ::core::option::Option<::planus::Vector<'a, self::BlockRef<'a>>>,
                        > {
                            self.0.access(3, "Footer", "record_batches")
                        }

                        /// Getter for the [`custom_metadata` field](Footer#structfield.custom_metadata).
                        #[inline]
                        pub fn custom_metadata(
                            &self,
                        ) -> ::planus::Result<
                            ::core::option::Option<
                                ::planus::Vector<'a, ::planus::Result<self::KeyValueRef<'a>>>,
                            >,
                        > {
                            self.0.access(4, "Footer", "custom_metadata")
                        }
                    }

                    impl<'a> ::core::fmt::Debug for FooterRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("FooterRef");
                            f.field("version", &self.version());
                            if let ::core::option::Option::Some(field_schema) =
                                self.schema().transpose()
                            {
                                f.field("schema", &field_schema);
                            }
                            if let ::core::option::Option::Some(field_dictionaries) =
                                self.dictionaries().transpose()
                            {
                                f.field("dictionaries", &field_dictionaries);
                            }
                            if let ::core::option::Option::Some(field_record_batches) =
                                self.record_batches().transpose()
                            {
                                f.field("record_batches", &field_record_batches);
                            }
                            if let ::core::option::Option::Some(field_custom_metadata) =
                                self.custom_metadata().transpose()
                            {
                                f.field("custom_metadata", &field_custom_metadata);
                            }
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<FooterRef<'a>> for Footer {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: FooterRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                version: ::core::convert::TryInto::try_into(value.version()?)?,
                                schema: if let ::core::option::Option::Some(schema) =
                                    value.schema()?
                                {
                                    ::core::option::Option::Some(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryInto::try_into(schema)?,
                                    ))
                                } else {
                                    ::core::option::Option::None
                                },
                                dictionaries: if let ::core::option::Option::Some(dictionaries) =
                                    value.dictionaries()?
                                {
                                    ::core::option::Option::Some(dictionaries.to_vec()?)
                                } else {
                                    ::core::option::Option::None
                                },
                                record_batches: if let ::core::option::Option::Some(
                                    record_batches,
                                ) = value.record_batches()?
                                {
                                    ::core::option::Option::Some(record_batches.to_vec()?)
                                } else {
                                    ::core::option::Option::None
                                },
                                custom_metadata: if let ::core::option::Option::Some(
                                    custom_metadata,
                                ) = value.custom_metadata()?
                                {
                                    ::core::option::Option::Some(custom_metadata.to_vec_result()?)
                                } else {
                                    ::core::option::Option::None
                                },
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for FooterRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for FooterRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[FooterRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<Footer>> for Footer {
                        type Value = ::planus::Offset<Footer>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Footer>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for FooterRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[FooterRef]", "read_as_root", 0)
                            })
                        }
                    }

                    /// The struct `Block` in the namespace `org.apache.arrow.flatbuf`
                    ///
                    /// Generated from these locations:
                    /// * Struct `Block` in the file `File.fbs:39`
                    #[derive(
                        Copy,
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        Default,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct Block {
                        ///  Index to the start of the RecordBlock (note this is past the Message header)
                        pub offset: i64,

                        ///  Length of the metadata
                        pub meta_data_length: i32,

                        ///  Length of the data (this is aligned so there can be a gap between this and
                        ///  the metadata).
                        pub body_length: i64,
                    }

                    /// # Safety
                    /// The Planus compiler correctly calculates `ALIGNMENT` and `SIZE`.
                    unsafe impl ::planus::Primitive for Block {
                        const ALIGNMENT: usize = 8;
                        const SIZE: usize = 24;
                    }

                    #[allow(clippy::identity_op)]
                    impl ::planus::WriteAsPrimitive<Block> for Block {
                        #[inline]
                        fn write<const N: usize>(
                            &self,
                            cursor: ::planus::Cursor<'_, N>,
                            buffer_position: u32,
                        ) {
                            let (cur, cursor) = cursor.split::<8, 16>();
                            self.offset.write(cur, buffer_position - 0);
                            let (cur, cursor) = cursor.split::<4, 12>();
                            self.meta_data_length.write(cur, buffer_position - 8);
                            let cursor = cursor.write::<4, 8>([0; 4]);
                            let (cur, cursor) = cursor.split::<8, 0>();
                            self.body_length.write(cur, buffer_position - 16);
                            cursor.finish([]);
                        }
                    }

                    impl ::planus::WriteAsOffset<Block> for Block {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Block> {
                            unsafe {
                                builder.write_with(24, 7, |buffer_position, bytes| {
                                    let bytes = bytes.as_mut_ptr();

                                    ::planus::WriteAsPrimitive::write(
                                        self,
                                        ::planus::Cursor::new(
                                            &mut *(bytes
                                                as *mut [::core::mem::MaybeUninit<u8>; 24]),
                                        ),
                                        buffer_position,
                                    );
                                });
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<Block> for Block {
                        type Prepared = Self;
                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self {
                            *self
                        }
                    }

                    impl ::planus::WriteAsOptional<Block> for Block {
                        type Prepared = Self;
                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<Self> {
                            ::core::option::Option::Some(*self)
                        }
                    }

                    /// Reference to a deserialized [Block].
                    #[derive(Copy, Clone)]
                    pub struct BlockRef<'a>(::planus::ArrayWithStartOffset<'a, 24>);

                    impl<'a> BlockRef<'a> {
                        /// Getter for the [`offset` field](Block#structfield.offset).
                        pub fn offset(&self) -> i64 {
                            let buffer = self.0.advance_as_array::<8>(0).unwrap();

                            i64::from_le_bytes(*buffer.as_array())
                        }

                        /// Getter for the [`metaDataLength` field](Block#structfield.meta_data_length).
                        pub fn meta_data_length(&self) -> i32 {
                            let buffer = self.0.advance_as_array::<4>(8).unwrap();

                            i32::from_le_bytes(*buffer.as_array())
                        }

                        /// Getter for the [`bodyLength` field](Block#structfield.body_length).
                        pub fn body_length(&self) -> i64 {
                            let buffer = self.0.advance_as_array::<8>(16).unwrap();

                            i64::from_le_bytes(*buffer.as_array())
                        }
                    }

                    impl<'a> ::core::fmt::Debug for BlockRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("BlockRef");
                            f.field("offset", &self.offset());
                            f.field("meta_data_length", &self.meta_data_length());
                            f.field("body_length", &self.body_length());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::From<::planus::ArrayWithStartOffset<'a, 24>> for BlockRef<'a> {
                        fn from(array: ::planus::ArrayWithStartOffset<'a, 24>) -> Self {
                            Self(array)
                        }
                    }

                    impl<'a> ::core::convert::From<BlockRef<'a>> for Block {
                        #[allow(unreachable_code)]
                        fn from(value: BlockRef<'a>) -> Self {
                            Self {
                                offset: value.offset(),
                                meta_data_length: value.meta_data_length(),
                                body_length: value.body_length(),
                            }
                        }
                    }

                    impl<'a, 'b> ::core::cmp::PartialEq<BlockRef<'a>> for BlockRef<'b> {
                        fn eq(&self, other: &BlockRef<'_>) -> bool {
                            self.offset() == other.offset()
                                && self.meta_data_length() == other.meta_data_length()
                                && self.body_length() == other.body_length()
                        }
                    }

                    impl<'a> ::core::cmp::Eq for BlockRef<'a> {}
                    impl<'a, 'b> ::core::cmp::PartialOrd<BlockRef<'a>> for BlockRef<'b> {
                        fn partial_cmp(
                            &self,
                            other: &BlockRef<'_>,
                        ) -> ::core::option::Option<::core::cmp::Ordering> {
                            ::core::option::Option::Some(::core::cmp::Ord::cmp(self, other))
                        }
                    }

                    impl<'a> ::core::cmp::Ord for BlockRef<'a> {
                        fn cmp(&self, other: &BlockRef<'_>) -> ::core::cmp::Ordering {
                            self.offset()
                                .cmp(&other.offset())
                                .then_with(|| {
                                    self.meta_data_length().cmp(&other.meta_data_length())
                                })
                                .then_with(|| self.body_length().cmp(&other.body_length()))
                        }
                    }

                    impl<'a> ::core::hash::Hash for BlockRef<'a> {
                        fn hash<H: ::core::hash::Hasher>(&self, state: &mut H) {
                            self.offset().hash(state);
                            self.meta_data_length().hash(state);
                            self.body_length().hash(state);
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for BlockRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            let buffer = buffer.advance_as_array::<24>(offset)?;
                            ::core::result::Result::Ok(Self(buffer))
                        }
                    }

                    impl<'a> ::planus::VectorRead<'a> for BlockRef<'a> {
                        const STRIDE: usize = 24;

                        #[inline]
                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> Self {
                            Self(unsafe { buffer.unchecked_advance_as_array(offset) })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<Block> for Block {
                        const STRIDE: usize = 24;

                        type Value = Block;

                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self::Value {
                            *self
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[Block],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 24];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (24 * i) as u32,
                                );
                            }
                        }
                    }

                    /// The enum `MetadataVersion` in the namespace `org.apache.arrow.flatbuf`
                    ///
                    /// Generated from these locations:
                    /// * Enum `MetadataVersion` in the file `Schema.fbs:31`
                    #[derive(
                        Copy,
                        Clone,
                        Debug,
                        PartialEq,
                        Eq,
                        PartialOrd,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    #[repr(i16)]
                    pub enum MetadataVersion {
                        ///  0.1.0 (October 2016).
                        V1 = 0,

                        ///  0.2.0 (February 2017). Non-backwards compatible with V1.
                        V2 = 1,

                        ///  0.3.0 -> 0.7.1 (May - December 2017). Non-backwards compatible with V2.
                        V3 = 2,

                        ///  >= 0.8.0 (December 2017). Non-backwards compatible with V3.
                        V4 = 3,

                        ///  >= 1.0.0 (July 2020). Backwards compatible with V4 (V5 readers can read V4
                        ///  metadata and IPC messages). Implementations are recommended to provide a
                        ///  V4 compatibility mode with V5 format changes disabled.
                        ///
                        ///  Incompatible changes between V4 and V5:
                        ///  - Union buffer layout has changed. In V5, Unions don't have a validity
                        ///    bitmap buffer.
                        V5 = 4,
                    }

                    impl MetadataVersion {
                        /// Array containing all valid variants of MetadataVersion
                        pub const ENUM_VALUES: [Self; 5] =
                            [Self::V1, Self::V2, Self::V3, Self::V4, Self::V5];
                    }

                    impl ::core::convert::TryFrom<i16> for MetadataVersion {
                        type Error = ::planus::errors::UnknownEnumTagKind;
                        #[inline]
                        fn try_from(
                            value: i16,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTagKind>
                        {
                            #[allow(clippy::match_single_binding)]
                            match value {
                                0 => ::core::result::Result::Ok(MetadataVersion::V1),
                                1 => ::core::result::Result::Ok(MetadataVersion::V2),
                                2 => ::core::result::Result::Ok(MetadataVersion::V3),
                                3 => ::core::result::Result::Ok(MetadataVersion::V4),
                                4 => ::core::result::Result::Ok(MetadataVersion::V5),

                                _ => ::core::result::Result::Err(
                                    ::planus::errors::UnknownEnumTagKind { tag: value as i128 },
                                ),
                            }
                        }
                    }

                    impl ::core::convert::From<MetadataVersion> for i16 {
                        #[inline]
                        fn from(value: MetadataVersion) -> Self {
                            value as i16
                        }
                    }

                    /// # Safety
                    /// The Planus compiler correctly calculates `ALIGNMENT` and `SIZE`.
                    unsafe impl ::planus::Primitive for MetadataVersion {
                        const ALIGNMENT: usize = 2;
                        const SIZE: usize = 2;
                    }

                    impl ::planus::WriteAsPrimitive<MetadataVersion> for MetadataVersion {
                        #[inline]
                        fn write<const N: usize>(
                            &self,
                            cursor: ::planus::Cursor<'_, N>,
                            buffer_position: u32,
                        ) {
                            (*self as i16).write(cursor, buffer_position);
                        }
                    }

                    impl ::planus::WriteAs<MetadataVersion> for MetadataVersion {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> MetadataVersion {
                            *self
                        }
                    }

                    impl ::planus::WriteAsDefault<MetadataVersion, MetadataVersion> for MetadataVersion {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                            default: &MetadataVersion,
                        ) -> ::core::option::Option<MetadataVersion> {
                            if self == default {
                                ::core::option::Option::None
                            } else {
                                ::core::option::Option::Some(*self)
                            }
                        }
                    }

                    impl ::planus::WriteAsOptional<MetadataVersion> for MetadataVersion {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<MetadataVersion> {
                            ::core::option::Option::Some(*self)
                        }
                    }

                    impl<'buf> ::planus::TableRead<'buf> for MetadataVersion {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            let n: i16 = ::planus::TableRead::from_buffer(buffer, offset)?;
                            ::core::result::Result::Ok(::core::convert::TryInto::try_into(n)?)
                        }
                    }

                    impl<'buf> ::planus::VectorReadInner<'buf> for MetadataVersion {
                        type Error = ::planus::errors::UnknownEnumTag;
                        const STRIDE: usize = 2;
                        #[inline]
                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTag>
                        {
                            let value = unsafe {
                                <i16 as ::planus::VectorRead>::from_buffer(buffer, offset)
                            };
                            let value: ::core::result::Result<Self, _> =
                                ::core::convert::TryInto::try_into(value);
                            value.map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "MetadataVersion",
                                    "VectorRead::from_buffer",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<MetadataVersion> for MetadataVersion {
                        const STRIDE: usize = 2;

                        type Value = Self;

                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self {
                            *self
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[Self],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 2];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (2 * i) as u32,
                                );
                            }
                        }
                    }

                    ///  Represents Arrow Features that might not have full support
                    ///  within implementations. This is intended to be used in
                    ///  two scenarios:
                    ///   1.  A mechanism for readers of Arrow Streams
                    ///       and files to understand that the stream or file makes
                    ///       use of a feature that isn't supported or unknown to
                    ///       the implementation (and therefore can meet the Arrow
                    ///       forward compatibility guarantees).
                    ///   2.  A means of negotiating between a client and server
                    ///       what features a stream is allowed to use. The enums
                    ///       values here are intended to represent higher level
                    ///       features, additional details may be negotiated
                    ///       with key-value pairs specific to the protocol.
                    ///
                    ///  Enums added to this list should be assigned power-of-two values
                    ///  to facilitate exchanging and comparing bitmaps for supported
                    ///  features.
                    ///
                    /// Generated from these locations:
                    /// * Enum `Feature` in the file `Schema.fbs:71`
                    #[derive(
                        Copy,
                        Clone,
                        Debug,
                        PartialEq,
                        Eq,
                        PartialOrd,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    #[repr(i64)]
                    pub enum Feature {
                        ///  Needed to make flatbuffers happy.
                        Unused = 0,

                        ///  The stream makes use of multiple full dictionaries with the
                        ///  same ID and assumes clients implement dictionary replacement
                        ///  correctly.
                        DictionaryReplacement = 1,

                        ///  The stream makes use of compressed bodies as described
                        ///  in Message.fbs.
                        CompressedBody = 2,
                    }

                    impl Feature {
                        /// Array containing all valid variants of Feature
                        pub const ENUM_VALUES: [Self; 3] = [
                            Self::Unused,
                            Self::DictionaryReplacement,
                            Self::CompressedBody,
                        ];
                    }

                    impl ::core::convert::TryFrom<i64> for Feature {
                        type Error = ::planus::errors::UnknownEnumTagKind;
                        #[inline]
                        fn try_from(
                            value: i64,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTagKind>
                        {
                            #[allow(clippy::match_single_binding)]
                            match value {
                                0 => ::core::result::Result::Ok(Feature::Unused),
                                1 => ::core::result::Result::Ok(Feature::DictionaryReplacement),
                                2 => ::core::result::Result::Ok(Feature::CompressedBody),

                                _ => ::core::result::Result::Err(
                                    ::planus::errors::UnknownEnumTagKind { tag: value as i128 },
                                ),
                            }
                        }
                    }

                    impl ::core::convert::From<Feature> for i64 {
                        #[inline]
                        fn from(value: Feature) -> Self {
                            value as i64
                        }
                    }

                    /// # Safety
                    /// The Planus compiler correctly calculates `ALIGNMENT` and `SIZE`.
                    unsafe impl ::planus::Primitive for Feature {
                        const ALIGNMENT: usize = 8;
                        const SIZE: usize = 8;
                    }

                    impl ::planus::WriteAsPrimitive<Feature> for Feature {
                        #[inline]
                        fn write<const N: usize>(
                            &self,
                            cursor: ::planus::Cursor<'_, N>,
                            buffer_position: u32,
                        ) {
                            (*self as i64).write(cursor, buffer_position);
                        }
                    }

                    impl ::planus::WriteAs<Feature> for Feature {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Feature {
                            *self
                        }
                    }

                    impl ::planus::WriteAsDefault<Feature, Feature> for Feature {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                            default: &Feature,
                        ) -> ::core::option::Option<Feature> {
                            if self == default {
                                ::core::option::Option::None
                            } else {
                                ::core::option::Option::Some(*self)
                            }
                        }
                    }

                    impl ::planus::WriteAsOptional<Feature> for Feature {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<Feature> {
                            ::core::option::Option::Some(*self)
                        }
                    }

                    impl<'buf> ::planus::TableRead<'buf> for Feature {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            let n: i64 = ::planus::TableRead::from_buffer(buffer, offset)?;
                            ::core::result::Result::Ok(::core::convert::TryInto::try_into(n)?)
                        }
                    }

                    impl<'buf> ::planus::VectorReadInner<'buf> for Feature {
                        type Error = ::planus::errors::UnknownEnumTag;
                        const STRIDE: usize = 8;
                        #[inline]
                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTag>
                        {
                            let value = unsafe {
                                <i64 as ::planus::VectorRead>::from_buffer(buffer, offset)
                            };
                            let value: ::core::result::Result<Self, _> =
                                ::core::convert::TryInto::try_into(value);
                            value.map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "Feature",
                                    "VectorRead::from_buffer",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<Feature> for Feature {
                        const STRIDE: usize = 8;

                        type Value = Self;

                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self {
                            *self
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[Self],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 8];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (8 * i) as u32,
                                );
                            }
                        }
                    }

                    ///  These are stored in the flatbuffer in the Type union below
                    ///
                    /// Generated from these locations:
                    /// * Table `Null` in the file `Schema.fbs:85`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct Null {}

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for Null {
                        fn default() -> Self {
                            Self {}
                        }
                    }

                    impl Null {
                        /// Creates a [NullBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> NullBuilder<()> {
                            NullBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(builder: &mut ::planus::Builder) -> ::planus::Offset<Self> {
                            let table_writer: ::planus::table_writer::TableWriter<4> =
                                ::core::default::Default::default();
                            unsafe {
                                table_writer.finish(builder, |_table_writer| {});
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Null>> for Null {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Null> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Null>> for Null {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Null>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Null> for Null {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Null> {
                            Null::create(builder)
                        }
                    }

                    /// Builder for serializing an instance of the [Null] type.
                    ///
                    /// Can be created using the [Null::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct NullBuilder<State>(State);

                    impl NullBuilder<()> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Null].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Null>
                        where
                            Self: ::planus::WriteAsOffset<Null>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Null>> for NullBuilder<()> {
                        type Prepared = ::planus::Offset<Null>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Null> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Null>> for NullBuilder<()> {
                        type Prepared = ::planus::Offset<Null>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Null>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Null> for NullBuilder<()> {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Null> {
                            Null::create(builder)
                        }
                    }

                    /// Reference to a deserialized [Null].
                    #[derive(Copy, Clone)]
                    pub struct NullRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> NullRef<'a> {}

                    impl<'a> ::core::fmt::Debug for NullRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("NullRef");

                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<NullRef<'a>> for Null {
                        type Error = ::planus::Error;

                        fn try_from(_value: NullRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {})
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for NullRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for NullRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[NullRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<Null>> for Null {
                        type Value = ::planus::Offset<Null>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Null>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for NullRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[NullRef]", "read_as_root", 0)
                            })
                        }
                    }

                    ///  A Struct_ in the flatbuffer metadata is the same as an Arrow Struct
                    ///  (according to the physical memory layout). We used Struct_ here as
                    ///  Struct is a reserved word in Flatbuffers
                    ///
                    /// Generated from these locations:
                    /// * Table `Struct_` in the file `Schema.fbs:91`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct Struct {}

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for Struct {
                        fn default() -> Self {
                            Self {}
                        }
                    }

                    impl Struct {
                        /// Creates a [StructBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> StructBuilder<()> {
                            StructBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(builder: &mut ::planus::Builder) -> ::planus::Offset<Self> {
                            let table_writer: ::planus::table_writer::TableWriter<4> =
                                ::core::default::Default::default();
                            unsafe {
                                table_writer.finish(builder, |_table_writer| {});
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Struct>> for Struct {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Struct> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Struct>> for Struct {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Struct>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Struct> for Struct {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Struct> {
                            Struct::create(builder)
                        }
                    }

                    /// Builder for serializing an instance of the [Struct] type.
                    ///
                    /// Can be created using the [Struct::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct StructBuilder<State>(State);

                    impl StructBuilder<()> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Struct].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Struct>
                        where
                            Self: ::planus::WriteAsOffset<Struct>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Struct>> for StructBuilder<()> {
                        type Prepared = ::planus::Offset<Struct>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Struct> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Struct>> for StructBuilder<()> {
                        type Prepared = ::planus::Offset<Struct>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Struct>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Struct> for StructBuilder<()> {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Struct> {
                            Struct::create(builder)
                        }
                    }

                    /// Reference to a deserialized [Struct].
                    #[derive(Copy, Clone)]
                    pub struct StructRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> StructRef<'a> {}

                    impl<'a> ::core::fmt::Debug for StructRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("StructRef");

                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<StructRef<'a>> for Struct {
                        type Error = ::planus::Error;

                        fn try_from(_value: StructRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {})
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for StructRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for StructRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[StructRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<Struct>> for Struct {
                        type Value = ::planus::Offset<Struct>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Struct>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for StructRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[StructRef]", "read_as_root", 0)
                            })
                        }
                    }

                    /// The table `List` in the namespace `org.apache.arrow.flatbuf`
                    ///
                    /// Generated from these locations:
                    /// * Table `List` in the file `Schema.fbs:94`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct List {}

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for List {
                        fn default() -> Self {
                            Self {}
                        }
                    }

                    impl List {
                        /// Creates a [ListBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> ListBuilder<()> {
                            ListBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(builder: &mut ::planus::Builder) -> ::planus::Offset<Self> {
                            let table_writer: ::planus::table_writer::TableWriter<4> =
                                ::core::default::Default::default();
                            unsafe {
                                table_writer.finish(builder, |_table_writer| {});
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<List>> for List {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<List> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<List>> for List {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<List>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<List> for List {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<List> {
                            List::create(builder)
                        }
                    }

                    /// Builder for serializing an instance of the [List] type.
                    ///
                    /// Can be created using the [List::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct ListBuilder<State>(State);

                    impl ListBuilder<()> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [List].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<List>
                        where
                            Self: ::planus::WriteAsOffset<List>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<List>> for ListBuilder<()> {
                        type Prepared = ::planus::Offset<List>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<List> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<List>> for ListBuilder<()> {
                        type Prepared = ::planus::Offset<List>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<List>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<List> for ListBuilder<()> {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<List> {
                            List::create(builder)
                        }
                    }

                    /// Reference to a deserialized [List].
                    #[derive(Copy, Clone)]
                    pub struct ListRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> ListRef<'a> {}

                    impl<'a> ::core::fmt::Debug for ListRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("ListRef");

                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<ListRef<'a>> for List {
                        type Error = ::planus::Error;

                        fn try_from(_value: ListRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {})
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for ListRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for ListRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[ListRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<List>> for List {
                        type Value = ::planus::Offset<List>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<List>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for ListRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[ListRef]", "read_as_root", 0)
                            })
                        }
                    }

                    ///  Same as List, but with 64-bit offsets, allowing to represent
                    ///  extremely large data values.
                    ///
                    /// Generated from these locations:
                    /// * Table `LargeList` in the file `Schema.fbs:99`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct LargeList {}

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for LargeList {
                        fn default() -> Self {
                            Self {}
                        }
                    }

                    impl LargeList {
                        /// Creates a [LargeListBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> LargeListBuilder<()> {
                            LargeListBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(builder: &mut ::planus::Builder) -> ::planus::Offset<Self> {
                            let table_writer: ::planus::table_writer::TableWriter<4> =
                                ::core::default::Default::default();
                            unsafe {
                                table_writer.finish(builder, |_table_writer| {});
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<LargeList>> for LargeList {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<LargeList> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<LargeList>> for LargeList {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<LargeList>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<LargeList> for LargeList {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<LargeList> {
                            LargeList::create(builder)
                        }
                    }

                    /// Builder for serializing an instance of the [LargeList] type.
                    ///
                    /// Can be created using the [LargeList::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct LargeListBuilder<State>(State);

                    impl LargeListBuilder<()> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [LargeList].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<LargeList>
                        where
                            Self: ::planus::WriteAsOffset<LargeList>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<LargeList>> for LargeListBuilder<()> {
                        type Prepared = ::planus::Offset<LargeList>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<LargeList> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<LargeList>> for LargeListBuilder<()> {
                        type Prepared = ::planus::Offset<LargeList>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<LargeList>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<LargeList> for LargeListBuilder<()> {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<LargeList> {
                            LargeList::create(builder)
                        }
                    }

                    /// Reference to a deserialized [LargeList].
                    #[derive(Copy, Clone)]
                    pub struct LargeListRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> LargeListRef<'a> {}

                    impl<'a> ::core::fmt::Debug for LargeListRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("LargeListRef");

                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<LargeListRef<'a>> for LargeList {
                        type Error = ::planus::Error;

                        fn try_from(_value: LargeListRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {})
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for LargeListRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for LargeListRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[LargeListRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<LargeList>> for LargeList {
                        type Value = ::planus::Offset<LargeList>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<LargeList>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for LargeListRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[LargeListRef]", "read_as_root", 0)
                            })
                        }
                    }

                    ///  Represents the same logical types that List can, but contains offsets and
                    ///  sizes allowing for writes in any order and sharing of child values among
                    ///  list values.
                    ///
                    /// Generated from these locations:
                    /// * Table `ListView` in the file `Schema.fbs:105`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct ListView {}

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for ListView {
                        fn default() -> Self {
                            Self {}
                        }
                    }

                    impl ListView {
                        /// Creates a [ListViewBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> ListViewBuilder<()> {
                            ListViewBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(builder: &mut ::planus::Builder) -> ::planus::Offset<Self> {
                            let table_writer: ::planus::table_writer::TableWriter<4> =
                                ::core::default::Default::default();
                            unsafe {
                                table_writer.finish(builder, |_table_writer| {});
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<ListView>> for ListView {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<ListView> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<ListView>> for ListView {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<ListView>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<ListView> for ListView {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<ListView> {
                            ListView::create(builder)
                        }
                    }

                    /// Builder for serializing an instance of the [ListView] type.
                    ///
                    /// Can be created using the [ListView::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct ListViewBuilder<State>(State);

                    impl ListViewBuilder<()> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [ListView].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<ListView>
                        where
                            Self: ::planus::WriteAsOffset<ListView>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<ListView>> for ListViewBuilder<()> {
                        type Prepared = ::planus::Offset<ListView>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<ListView> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<ListView>> for ListViewBuilder<()> {
                        type Prepared = ::planus::Offset<ListView>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<ListView>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<ListView> for ListViewBuilder<()> {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<ListView> {
                            ListView::create(builder)
                        }
                    }

                    /// Reference to a deserialized [ListView].
                    #[derive(Copy, Clone)]
                    pub struct ListViewRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> ListViewRef<'a> {}

                    impl<'a> ::core::fmt::Debug for ListViewRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("ListViewRef");

                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<ListViewRef<'a>> for ListView {
                        type Error = ::planus::Error;

                        fn try_from(_value: ListViewRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {})
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for ListViewRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for ListViewRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[ListViewRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<ListView>> for ListView {
                        type Value = ::planus::Offset<ListView>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<ListView>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for ListViewRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[ListViewRef]", "read_as_root", 0)
                            })
                        }
                    }

                    ///  Same as ListView, but with 64-bit offsets and sizes, allowing to represent
                    ///  extremely large data values.
                    ///
                    /// Generated from these locations:
                    /// * Table `LargeListView` in the file `Schema.fbs:110`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct LargeListView {}

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for LargeListView {
                        fn default() -> Self {
                            Self {}
                        }
                    }

                    impl LargeListView {
                        /// Creates a [LargeListViewBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> LargeListViewBuilder<()> {
                            LargeListViewBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(builder: &mut ::planus::Builder) -> ::planus::Offset<Self> {
                            let table_writer: ::planus::table_writer::TableWriter<4> =
                                ::core::default::Default::default();
                            unsafe {
                                table_writer.finish(builder, |_table_writer| {});
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<LargeListView>> for LargeListView {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<LargeListView> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<LargeListView>> for LargeListView {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<LargeListView>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<LargeListView> for LargeListView {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<LargeListView> {
                            LargeListView::create(builder)
                        }
                    }

                    /// Builder for serializing an instance of the [LargeListView] type.
                    ///
                    /// Can be created using the [LargeListView::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct LargeListViewBuilder<State>(State);

                    impl LargeListViewBuilder<()> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [LargeListView].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<LargeListView>
                        where
                            Self: ::planus::WriteAsOffset<LargeListView>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<LargeListView>> for LargeListViewBuilder<()> {
                        type Prepared = ::planus::Offset<LargeListView>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<LargeListView> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<LargeListView>> for LargeListViewBuilder<()> {
                        type Prepared = ::planus::Offset<LargeListView>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<LargeListView>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<LargeListView> for LargeListViewBuilder<()> {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<LargeListView> {
                            LargeListView::create(builder)
                        }
                    }

                    /// Reference to a deserialized [LargeListView].
                    #[derive(Copy, Clone)]
                    pub struct LargeListViewRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> LargeListViewRef<'a> {}

                    impl<'a> ::core::fmt::Debug for LargeListViewRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("LargeListViewRef");

                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<LargeListViewRef<'a>> for LargeListView {
                        type Error = ::planus::Error;

                        fn try_from(_value: LargeListViewRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {})
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for LargeListViewRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for LargeListViewRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[LargeListViewRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<LargeListView>> for LargeListView {
                        type Value = ::planus::Offset<LargeListView>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<LargeListView>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for LargeListViewRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[LargeListViewRef]",
                                    "read_as_root",
                                    0,
                                )
                            })
                        }
                    }

                    /// The table `FixedSizeList` in the namespace `org.apache.arrow.flatbuf`
                    ///
                    /// Generated from these locations:
                    /// * Table `FixedSizeList` in the file `Schema.fbs:113`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct FixedSizeList {
                        ///  Number of list items per value
                        pub list_size: i32,
                    }

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for FixedSizeList {
                        fn default() -> Self {
                            Self { list_size: 0 }
                        }
                    }

                    impl FixedSizeList {
                        /// Creates a [FixedSizeListBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> FixedSizeListBuilder<()> {
                            FixedSizeListBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            field_list_size: impl ::planus::WriteAsDefault<i32, i32>,
                        ) -> ::planus::Offset<Self> {
                            let prepared_list_size = field_list_size.prepare(builder, &0);

                            let mut table_writer: ::planus::table_writer::TableWriter<6> =
                                ::core::default::Default::default();
                            if prepared_list_size.is_some() {
                                table_writer.write_entry::<i32>(0);
                            }

                            unsafe {
                                table_writer.finish(builder, |object_writer| {
                                    if let ::core::option::Option::Some(prepared_list_size) =
                                        prepared_list_size
                                    {
                                        object_writer.write::<_, _, 4>(&prepared_list_size);
                                    }
                                });
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<FixedSizeList>> for FixedSizeList {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<FixedSizeList> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<FixedSizeList>> for FixedSizeList {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<FixedSizeList>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<FixedSizeList> for FixedSizeList {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<FixedSizeList> {
                            FixedSizeList::create(builder, self.list_size)
                        }
                    }

                    /// Builder for serializing an instance of the [FixedSizeList] type.
                    ///
                    /// Can be created using the [FixedSizeList::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct FixedSizeListBuilder<State>(State);

                    impl FixedSizeListBuilder<()> {
                        /// Setter for the [`listSize` field](FixedSizeList#structfield.list_size).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn list_size<T0>(self, value: T0) -> FixedSizeListBuilder<(T0,)>
                        where
                            T0: ::planus::WriteAsDefault<i32, i32>,
                        {
                            FixedSizeListBuilder((value,))
                        }

                        /// Sets the [`listSize` field](FixedSizeList#structfield.list_size) to the default value.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn list_size_as_default(
                            self,
                        ) -> FixedSizeListBuilder<(::planus::DefaultValue,)>
                        {
                            self.list_size(::planus::DefaultValue)
                        }
                    }

                    impl<T0> FixedSizeListBuilder<(T0,)> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [FixedSizeList].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<FixedSizeList>
                        where
                            Self: ::planus::WriteAsOffset<FixedSizeList>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl<T0: ::planus::WriteAsDefault<i32, i32>>
                        ::planus::WriteAs<::planus::Offset<FixedSizeList>>
                        for FixedSizeListBuilder<(T0,)>
                    {
                        type Prepared = ::planus::Offset<FixedSizeList>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<FixedSizeList> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl<T0: ::planus::WriteAsDefault<i32, i32>>
                        ::planus::WriteAsOptional<::planus::Offset<FixedSizeList>>
                        for FixedSizeListBuilder<(T0,)>
                    {
                        type Prepared = ::planus::Offset<FixedSizeList>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<FixedSizeList>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl<T0: ::planus::WriteAsDefault<i32, i32>>
                        ::planus::WriteAsOffset<FixedSizeList> for FixedSizeListBuilder<(T0,)>
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<FixedSizeList> {
                            let (v0,) = &self.0;
                            FixedSizeList::create(builder, v0)
                        }
                    }

                    /// Reference to a deserialized [FixedSizeList].
                    #[derive(Copy, Clone)]
                    pub struct FixedSizeListRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> FixedSizeListRef<'a> {
                        /// Getter for the [`listSize` field](FixedSizeList#structfield.list_size).
                        #[inline]
                        pub fn list_size(&self) -> ::planus::Result<i32> {
                            ::core::result::Result::Ok(
                                self.0.access(0, "FixedSizeList", "list_size")?.unwrap_or(0),
                            )
                        }
                    }

                    impl<'a> ::core::fmt::Debug for FixedSizeListRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("FixedSizeListRef");
                            f.field("list_size", &self.list_size());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<FixedSizeListRef<'a>> for FixedSizeList {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: FixedSizeListRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                list_size: ::core::convert::TryInto::try_into(value.list_size()?)?,
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for FixedSizeListRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for FixedSizeListRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[FixedSizeListRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<FixedSizeList>> for FixedSizeList {
                        type Value = ::planus::Offset<FixedSizeList>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<FixedSizeList>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for FixedSizeListRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[FixedSizeListRef]",
                                    "read_as_root",
                                    0,
                                )
                            })
                        }
                    }

                    ///  A Map is a logical nested type that is represented as
                    ///
                    ///  List<entries: Struct<key: K, value: V>>
                    ///
                    ///  In this layout, the keys and values are each respectively contiguous. We do
                    ///  not constrain the key and value types, so the application is responsible
                    ///  for ensuring that the keys are hashable and unique. Whether the keys are sorted
                    ///  may be set in the metadata for this field.
                    ///
                    ///  In a field with Map type, the field has a child Struct field, which then
                    ///  has two children: key type and the second the value type. The names of the
                    ///  child fields may be respectively "entries", "key", and "value", but this is
                    ///  not enforced.
                    ///
                    ///  Map
                    ///  ```text
                    ///    - child[0] entries: Struct
                    ///      - child[0] key: K
                    ///      - child[1] value: V
                    ///  ```
                    ///  Neither the "entries" field nor the "key" field may be nullable.
                    ///
                    ///  The metadata is structured so that Arrow systems without special handling
                    ///  for Map can make Map an alias for List. The "layout" attribute for the Map
                    ///  field must have the same contents as a List.
                    ///
                    /// Generated from these locations:
                    /// * Table `Map` in the file `Schema.fbs:143`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct Map {
                        ///  Set to true if the keys within each value are sorted
                        pub keys_sorted: bool,
                    }

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for Map {
                        fn default() -> Self {
                            Self { keys_sorted: false }
                        }
                    }

                    impl Map {
                        /// Creates a [MapBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> MapBuilder<()> {
                            MapBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            field_keys_sorted: impl ::planus::WriteAsDefault<bool, bool>,
                        ) -> ::planus::Offset<Self> {
                            let prepared_keys_sorted = field_keys_sorted.prepare(builder, &false);

                            let mut table_writer: ::planus::table_writer::TableWriter<6> =
                                ::core::default::Default::default();
                            if prepared_keys_sorted.is_some() {
                                table_writer.write_entry::<bool>(0);
                            }

                            unsafe {
                                table_writer.finish(builder, |object_writer| {
                                    if let ::core::option::Option::Some(prepared_keys_sorted) =
                                        prepared_keys_sorted
                                    {
                                        object_writer.write::<_, _, 1>(&prepared_keys_sorted);
                                    }
                                });
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Map>> for Map {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Map> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Map>> for Map {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Map>> {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Map> for Map {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Map> {
                            Map::create(builder, self.keys_sorted)
                        }
                    }

                    /// Builder for serializing an instance of the [Map] type.
                    ///
                    /// Can be created using the [Map::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct MapBuilder<State>(State);

                    impl MapBuilder<()> {
                        /// Setter for the [`keysSorted` field](Map#structfield.keys_sorted).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn keys_sorted<T0>(self, value: T0) -> MapBuilder<(T0,)>
                        where
                            T0: ::planus::WriteAsDefault<bool, bool>,
                        {
                            MapBuilder((value,))
                        }

                        /// Sets the [`keysSorted` field](Map#structfield.keys_sorted) to the default value.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn keys_sorted_as_default(
                            self,
                        ) -> MapBuilder<(::planus::DefaultValue,)> {
                            self.keys_sorted(::planus::DefaultValue)
                        }
                    }

                    impl<T0> MapBuilder<(T0,)> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Map].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Map>
                        where
                            Self: ::planus::WriteAsOffset<Map>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl<T0: ::planus::WriteAsDefault<bool, bool>>
                        ::planus::WriteAs<::planus::Offset<Map>> for MapBuilder<(T0,)>
                    {
                        type Prepared = ::planus::Offset<Map>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Map> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl<T0: ::planus::WriteAsDefault<bool, bool>>
                        ::planus::WriteAsOptional<::planus::Offset<Map>> for MapBuilder<(T0,)>
                    {
                        type Prepared = ::planus::Offset<Map>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Map>> {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl<T0: ::planus::WriteAsDefault<bool, bool>> ::planus::WriteAsOffset<Map> for MapBuilder<(T0,)> {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Map> {
                            let (v0,) = &self.0;
                            Map::create(builder, v0)
                        }
                    }

                    /// Reference to a deserialized [Map].
                    #[derive(Copy, Clone)]
                    pub struct MapRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> MapRef<'a> {
                        /// Getter for the [`keysSorted` field](Map#structfield.keys_sorted).
                        #[inline]
                        pub fn keys_sorted(&self) -> ::planus::Result<bool> {
                            ::core::result::Result::Ok(
                                self.0.access(0, "Map", "keys_sorted")?.unwrap_or(false),
                            )
                        }
                    }

                    impl<'a> ::core::fmt::Debug for MapRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("MapRef");
                            f.field("keys_sorted", &self.keys_sorted());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<MapRef<'a>> for Map {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: MapRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                keys_sorted: ::core::convert::TryInto::try_into(
                                    value.keys_sorted()?,
                                )?,
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for MapRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for MapRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[MapRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<Map>> for Map {
                        type Value = ::planus::Offset<Map>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Map>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for MapRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[MapRef]", "read_as_root", 0)
                            })
                        }
                    }

                    /// The enum `UnionMode` in the namespace `org.apache.arrow.flatbuf`
                    ///
                    /// Generated from these locations:
                    /// * Enum `UnionMode` in the file `Schema.fbs:148`
                    #[derive(
                        Copy,
                        Clone,
                        Debug,
                        PartialEq,
                        Eq,
                        PartialOrd,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    #[repr(i16)]
                    pub enum UnionMode {
                        /// The variant `Sparse` in the enum `UnionMode`
                        Sparse = 0,

                        /// The variant `Dense` in the enum `UnionMode`
                        Dense = 1,
                    }

                    impl UnionMode {
                        /// Array containing all valid variants of UnionMode
                        pub const ENUM_VALUES: [Self; 2] = [Self::Sparse, Self::Dense];
                    }

                    impl ::core::convert::TryFrom<i16> for UnionMode {
                        type Error = ::planus::errors::UnknownEnumTagKind;
                        #[inline]
                        fn try_from(
                            value: i16,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTagKind>
                        {
                            #[allow(clippy::match_single_binding)]
                            match value {
                                0 => ::core::result::Result::Ok(UnionMode::Sparse),
                                1 => ::core::result::Result::Ok(UnionMode::Dense),

                                _ => ::core::result::Result::Err(
                                    ::planus::errors::UnknownEnumTagKind { tag: value as i128 },
                                ),
                            }
                        }
                    }

                    impl ::core::convert::From<UnionMode> for i16 {
                        #[inline]
                        fn from(value: UnionMode) -> Self {
                            value as i16
                        }
                    }

                    /// # Safety
                    /// The Planus compiler correctly calculates `ALIGNMENT` and `SIZE`.
                    unsafe impl ::planus::Primitive for UnionMode {
                        const ALIGNMENT: usize = 2;
                        const SIZE: usize = 2;
                    }

                    impl ::planus::WriteAsPrimitive<UnionMode> for UnionMode {
                        #[inline]
                        fn write<const N: usize>(
                            &self,
                            cursor: ::planus::Cursor<'_, N>,
                            buffer_position: u32,
                        ) {
                            (*self as i16).write(cursor, buffer_position);
                        }
                    }

                    impl ::planus::WriteAs<UnionMode> for UnionMode {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> UnionMode {
                            *self
                        }
                    }

                    impl ::planus::WriteAsDefault<UnionMode, UnionMode> for UnionMode {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                            default: &UnionMode,
                        ) -> ::core::option::Option<UnionMode> {
                            if self == default {
                                ::core::option::Option::None
                            } else {
                                ::core::option::Option::Some(*self)
                            }
                        }
                    }

                    impl ::planus::WriteAsOptional<UnionMode> for UnionMode {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<UnionMode> {
                            ::core::option::Option::Some(*self)
                        }
                    }

                    impl<'buf> ::planus::TableRead<'buf> for UnionMode {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            let n: i16 = ::planus::TableRead::from_buffer(buffer, offset)?;
                            ::core::result::Result::Ok(::core::convert::TryInto::try_into(n)?)
                        }
                    }

                    impl<'buf> ::planus::VectorReadInner<'buf> for UnionMode {
                        type Error = ::planus::errors::UnknownEnumTag;
                        const STRIDE: usize = 2;
                        #[inline]
                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTag>
                        {
                            let value = unsafe {
                                <i16 as ::planus::VectorRead>::from_buffer(buffer, offset)
                            };
                            let value: ::core::result::Result<Self, _> =
                                ::core::convert::TryInto::try_into(value);
                            value.map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "UnionMode",
                                    "VectorRead::from_buffer",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<UnionMode> for UnionMode {
                        const STRIDE: usize = 2;

                        type Value = Self;

                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self {
                            *self
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[Self],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 2];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (2 * i) as u32,
                                );
                            }
                        }
                    }

                    ///  A union is a complex type with children in Field
                    ///  By default ids in the type vector refer to the offsets in the children
                    ///  optionally typeIds provides an indirection between the child offset and the type id
                    ///  for each child `typeIds[offset]` is the id used in the type vector
                    ///
                    /// Generated from these locations:
                    /// * Table `Union` in the file `Schema.fbs:154`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct Union {
                        /// The field `mode` in the table `Union`
                        pub mode: self::UnionMode,
                        /// The field `typeIds` in the table `Union`
                        pub type_ids: ::core::option::Option<::planus::alloc::vec::Vec<i32>>,
                    }

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for Union {
                        fn default() -> Self {
                            Self {
                                mode: self::UnionMode::Sparse,
                                type_ids: ::core::default::Default::default(),
                            }
                        }
                    }

                    impl Union {
                        /// Creates a [UnionBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> UnionBuilder<()> {
                            UnionBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            field_mode: impl ::planus::WriteAsDefault<self::UnionMode, self::UnionMode>,
                            field_type_ids: impl ::planus::WriteAsOptional<::planus::Offset<[i32]>>,
                        ) -> ::planus::Offset<Self> {
                            let prepared_mode =
                                field_mode.prepare(builder, &self::UnionMode::Sparse);
                            let prepared_type_ids = field_type_ids.prepare(builder);

                            let mut table_writer: ::planus::table_writer::TableWriter<8> =
                                ::core::default::Default::default();
                            if prepared_type_ids.is_some() {
                                table_writer.write_entry::<::planus::Offset<[i32]>>(1);
                            }
                            if prepared_mode.is_some() {
                                table_writer.write_entry::<self::UnionMode>(0);
                            }

                            unsafe {
                                table_writer.finish(builder, |object_writer| {
                                    if let ::core::option::Option::Some(prepared_type_ids) =
                                        prepared_type_ids
                                    {
                                        object_writer.write::<_, _, 4>(&prepared_type_ids);
                                    }
                                    if let ::core::option::Option::Some(prepared_mode) =
                                        prepared_mode
                                    {
                                        object_writer.write::<_, _, 2>(&prepared_mode);
                                    }
                                });
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Union>> for Union {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Union> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Union>> for Union {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Union>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Union> for Union {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Union> {
                            Union::create(builder, self.mode, &self.type_ids)
                        }
                    }

                    /// Builder for serializing an instance of the [Union] type.
                    ///
                    /// Can be created using the [Union::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct UnionBuilder<State>(State);

                    impl UnionBuilder<()> {
                        /// Setter for the [`mode` field](Union#structfield.mode).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn mode<T0>(self, value: T0) -> UnionBuilder<(T0,)>
                        where
                            T0: ::planus::WriteAsDefault<self::UnionMode, self::UnionMode>,
                        {
                            UnionBuilder((value,))
                        }

                        /// Sets the [`mode` field](Union#structfield.mode) to the default value.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn mode_as_default(self) -> UnionBuilder<(::planus::DefaultValue,)> {
                            self.mode(::planus::DefaultValue)
                        }
                    }

                    impl<T0> UnionBuilder<(T0,)> {
                        /// Setter for the [`typeIds` field](Union#structfield.type_ids).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn type_ids<T1>(self, value: T1) -> UnionBuilder<(T0, T1)>
                        where
                            T1: ::planus::WriteAsOptional<::planus::Offset<[i32]>>,
                        {
                            let (v0,) = self.0;
                            UnionBuilder((v0, value))
                        }

                        /// Sets the [`typeIds` field](Union#structfield.type_ids) to null.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn type_ids_as_null(self) -> UnionBuilder<(T0, ())> {
                            self.type_ids(())
                        }
                    }

                    impl<T0, T1> UnionBuilder<(T0, T1)> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Union].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Union>
                        where
                            Self: ::planus::WriteAsOffset<Union>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<self::UnionMode, self::UnionMode>,
                            T1: ::planus::WriteAsOptional<::planus::Offset<[i32]>>,
                        > ::planus::WriteAs<::planus::Offset<Union>> for UnionBuilder<(T0, T1)>
                    {
                        type Prepared = ::planus::Offset<Union>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Union> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<self::UnionMode, self::UnionMode>,
                            T1: ::planus::WriteAsOptional<::planus::Offset<[i32]>>,
                        > ::planus::WriteAsOptional<::planus::Offset<Union>>
                        for UnionBuilder<(T0, T1)>
                    {
                        type Prepared = ::planus::Offset<Union>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Union>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<self::UnionMode, self::UnionMode>,
                            T1: ::planus::WriteAsOptional<::planus::Offset<[i32]>>,
                        > ::planus::WriteAsOffset<Union> for UnionBuilder<(T0, T1)>
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Union> {
                            let (v0, v1) = &self.0;
                            Union::create(builder, v0, v1)
                        }
                    }

                    /// Reference to a deserialized [Union].
                    #[derive(Copy, Clone)]
                    pub struct UnionRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> UnionRef<'a> {
                        /// Getter for the [`mode` field](Union#structfield.mode).
                        #[inline]
                        pub fn mode(&self) -> ::planus::Result<self::UnionMode> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(0, "Union", "mode")?
                                    .unwrap_or(self::UnionMode::Sparse),
                            )
                        }

                        /// Getter for the [`typeIds` field](Union#structfield.type_ids).
                        #[inline]
                        pub fn type_ids(
                            &self,
                        ) -> ::planus::Result<::core::option::Option<::planus::Vector<'a, i32>>>
                        {
                            self.0.access(1, "Union", "type_ids")
                        }
                    }

                    impl<'a> ::core::fmt::Debug for UnionRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("UnionRef");
                            f.field("mode", &self.mode());
                            if let ::core::option::Option::Some(field_type_ids) =
                                self.type_ids().transpose()
                            {
                                f.field("type_ids", &field_type_ids);
                            }
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<UnionRef<'a>> for Union {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: UnionRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                mode: ::core::convert::TryInto::try_into(value.mode()?)?,
                                type_ids: if let ::core::option::Option::Some(type_ids) =
                                    value.type_ids()?
                                {
                                    ::core::option::Option::Some(type_ids.to_vec()?)
                                } else {
                                    ::core::option::Option::None
                                },
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for UnionRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for UnionRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[UnionRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<Union>> for Union {
                        type Value = ::planus::Offset<Union>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Union>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for UnionRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[UnionRef]", "read_as_root", 0)
                            })
                        }
                    }

                    /// The table `Int` in the namespace `org.apache.arrow.flatbuf`
                    ///
                    /// Generated from these locations:
                    /// * Table `Int` in the file `Schema.fbs:159`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct Int {
                        /// The field `bitWidth` in the table `Int`
                        pub bit_width: i32,
                        /// The field `is_signed` in the table `Int`
                        pub is_signed: bool,
                    }

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for Int {
                        fn default() -> Self {
                            Self {
                                bit_width: 0,
                                is_signed: false,
                            }
                        }
                    }

                    impl Int {
                        /// Creates a [IntBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> IntBuilder<()> {
                            IntBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            field_bit_width: impl ::planus::WriteAsDefault<i32, i32>,
                            field_is_signed: impl ::planus::WriteAsDefault<bool, bool>,
                        ) -> ::planus::Offset<Self> {
                            let prepared_bit_width = field_bit_width.prepare(builder, &0);
                            let prepared_is_signed = field_is_signed.prepare(builder, &false);

                            let mut table_writer: ::planus::table_writer::TableWriter<8> =
                                ::core::default::Default::default();
                            if prepared_bit_width.is_some() {
                                table_writer.write_entry::<i32>(0);
                            }
                            if prepared_is_signed.is_some() {
                                table_writer.write_entry::<bool>(1);
                            }

                            unsafe {
                                table_writer.finish(builder, |object_writer| {
                                    if let ::core::option::Option::Some(prepared_bit_width) =
                                        prepared_bit_width
                                    {
                                        object_writer.write::<_, _, 4>(&prepared_bit_width);
                                    }
                                    if let ::core::option::Option::Some(prepared_is_signed) =
                                        prepared_is_signed
                                    {
                                        object_writer.write::<_, _, 1>(&prepared_is_signed);
                                    }
                                });
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Int>> for Int {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Int> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Int>> for Int {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Int>> {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Int> for Int {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Int> {
                            Int::create(builder, self.bit_width, self.is_signed)
                        }
                    }

                    /// Builder for serializing an instance of the [Int] type.
                    ///
                    /// Can be created using the [Int::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct IntBuilder<State>(State);

                    impl IntBuilder<()> {
                        /// Setter for the [`bitWidth` field](Int#structfield.bit_width).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn bit_width<T0>(self, value: T0) -> IntBuilder<(T0,)>
                        where
                            T0: ::planus::WriteAsDefault<i32, i32>,
                        {
                            IntBuilder((value,))
                        }

                        /// Sets the [`bitWidth` field](Int#structfield.bit_width) to the default value.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn bit_width_as_default(self) -> IntBuilder<(::planus::DefaultValue,)> {
                            self.bit_width(::planus::DefaultValue)
                        }
                    }

                    impl<T0> IntBuilder<(T0,)> {
                        /// Setter for the [`is_signed` field](Int#structfield.is_signed).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn is_signed<T1>(self, value: T1) -> IntBuilder<(T0, T1)>
                        where
                            T1: ::planus::WriteAsDefault<bool, bool>,
                        {
                            let (v0,) = self.0;
                            IntBuilder((v0, value))
                        }

                        /// Sets the [`is_signed` field](Int#structfield.is_signed) to the default value.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn is_signed_as_default(
                            self,
                        ) -> IntBuilder<(T0, ::planus::DefaultValue)> {
                            self.is_signed(::planus::DefaultValue)
                        }
                    }

                    impl<T0, T1> IntBuilder<(T0, T1)> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Int].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Int>
                        where
                            Self: ::planus::WriteAsOffset<Int>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<i32, i32>,
                            T1: ::planus::WriteAsDefault<bool, bool>,
                        > ::planus::WriteAs<::planus::Offset<Int>> for IntBuilder<(T0, T1)>
                    {
                        type Prepared = ::planus::Offset<Int>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Int> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<i32, i32>,
                            T1: ::planus::WriteAsDefault<bool, bool>,
                        > ::planus::WriteAsOptional<::planus::Offset<Int>>
                        for IntBuilder<(T0, T1)>
                    {
                        type Prepared = ::planus::Offset<Int>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Int>> {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<i32, i32>,
                            T1: ::planus::WriteAsDefault<bool, bool>,
                        > ::planus::WriteAsOffset<Int> for IntBuilder<(T0, T1)>
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Int> {
                            let (v0, v1) = &self.0;
                            Int::create(builder, v0, v1)
                        }
                    }

                    /// Reference to a deserialized [Int].
                    #[derive(Copy, Clone)]
                    pub struct IntRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> IntRef<'a> {
                        /// Getter for the [`bitWidth` field](Int#structfield.bit_width).
                        #[inline]
                        pub fn bit_width(&self) -> ::planus::Result<i32> {
                            ::core::result::Result::Ok(
                                self.0.access(0, "Int", "bit_width")?.unwrap_or(0),
                            )
                        }

                        /// Getter for the [`is_signed` field](Int#structfield.is_signed).
                        #[inline]
                        pub fn is_signed(&self) -> ::planus::Result<bool> {
                            ::core::result::Result::Ok(
                                self.0.access(1, "Int", "is_signed")?.unwrap_or(false),
                            )
                        }
                    }

                    impl<'a> ::core::fmt::Debug for IntRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("IntRef");
                            f.field("bit_width", &self.bit_width());
                            f.field("is_signed", &self.is_signed());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<IntRef<'a>> for Int {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: IntRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                bit_width: ::core::convert::TryInto::try_into(value.bit_width()?)?,
                                is_signed: ::core::convert::TryInto::try_into(value.is_signed()?)?,
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for IntRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for IntRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[IntRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<Int>> for Int {
                        type Value = ::planus::Offset<Int>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Int>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for IntRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[IntRef]", "read_as_root", 0)
                            })
                        }
                    }

                    /// The enum `Precision` in the namespace `org.apache.arrow.flatbuf`
                    ///
                    /// Generated from these locations:
                    /// * Enum `Precision` in the file `Schema.fbs:164`
                    #[derive(
                        Copy,
                        Clone,
                        Debug,
                        PartialEq,
                        Eq,
                        PartialOrd,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    #[repr(i16)]
                    pub enum Precision {
                        /// The variant `HALF` in the enum `Precision`
                        Half = 0,

                        /// The variant `SINGLE` in the enum `Precision`
                        Single = 1,

                        /// The variant `DOUBLE` in the enum `Precision`
                        Double = 2,
                    }

                    impl Precision {
                        /// Array containing all valid variants of Precision
                        pub const ENUM_VALUES: [Self; 3] = [Self::Half, Self::Single, Self::Double];
                    }

                    impl ::core::convert::TryFrom<i16> for Precision {
                        type Error = ::planus::errors::UnknownEnumTagKind;
                        #[inline]
                        fn try_from(
                            value: i16,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTagKind>
                        {
                            #[allow(clippy::match_single_binding)]
                            match value {
                                0 => ::core::result::Result::Ok(Precision::Half),
                                1 => ::core::result::Result::Ok(Precision::Single),
                                2 => ::core::result::Result::Ok(Precision::Double),

                                _ => ::core::result::Result::Err(
                                    ::planus::errors::UnknownEnumTagKind { tag: value as i128 },
                                ),
                            }
                        }
                    }

                    impl ::core::convert::From<Precision> for i16 {
                        #[inline]
                        fn from(value: Precision) -> Self {
                            value as i16
                        }
                    }

                    /// # Safety
                    /// The Planus compiler correctly calculates `ALIGNMENT` and `SIZE`.
                    unsafe impl ::planus::Primitive for Precision {
                        const ALIGNMENT: usize = 2;
                        const SIZE: usize = 2;
                    }

                    impl ::planus::WriteAsPrimitive<Precision> for Precision {
                        #[inline]
                        fn write<const N: usize>(
                            &self,
                            cursor: ::planus::Cursor<'_, N>,
                            buffer_position: u32,
                        ) {
                            (*self as i16).write(cursor, buffer_position);
                        }
                    }

                    impl ::planus::WriteAs<Precision> for Precision {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Precision {
                            *self
                        }
                    }

                    impl ::planus::WriteAsDefault<Precision, Precision> for Precision {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                            default: &Precision,
                        ) -> ::core::option::Option<Precision> {
                            if self == default {
                                ::core::option::Option::None
                            } else {
                                ::core::option::Option::Some(*self)
                            }
                        }
                    }

                    impl ::planus::WriteAsOptional<Precision> for Precision {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<Precision> {
                            ::core::option::Option::Some(*self)
                        }
                    }

                    impl<'buf> ::planus::TableRead<'buf> for Precision {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            let n: i16 = ::planus::TableRead::from_buffer(buffer, offset)?;
                            ::core::result::Result::Ok(::core::convert::TryInto::try_into(n)?)
                        }
                    }

                    impl<'buf> ::planus::VectorReadInner<'buf> for Precision {
                        type Error = ::planus::errors::UnknownEnumTag;
                        const STRIDE: usize = 2;
                        #[inline]
                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTag>
                        {
                            let value = unsafe {
                                <i16 as ::planus::VectorRead>::from_buffer(buffer, offset)
                            };
                            let value: ::core::result::Result<Self, _> =
                                ::core::convert::TryInto::try_into(value);
                            value.map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "Precision",
                                    "VectorRead::from_buffer",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<Precision> for Precision {
                        const STRIDE: usize = 2;

                        type Value = Self;

                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self {
                            *self
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[Self],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 2];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (2 * i) as u32,
                                );
                            }
                        }
                    }

                    /// The table `FloatingPoint` in the namespace `org.apache.arrow.flatbuf`
                    ///
                    /// Generated from these locations:
                    /// * Table `FloatingPoint` in the file `Schema.fbs:166`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct FloatingPoint {
                        /// The field `precision` in the table `FloatingPoint`
                        pub precision: self::Precision,
                    }

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for FloatingPoint {
                        fn default() -> Self {
                            Self {
                                precision: self::Precision::Half,
                            }
                        }
                    }

                    impl FloatingPoint {
                        /// Creates a [FloatingPointBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> FloatingPointBuilder<()> {
                            FloatingPointBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            field_precision: impl ::planus::WriteAsDefault<
                                self::Precision,
                                self::Precision,
                            >,
                        ) -> ::planus::Offset<Self> {
                            let prepared_precision =
                                field_precision.prepare(builder, &self::Precision::Half);

                            let mut table_writer: ::planus::table_writer::TableWriter<6> =
                                ::core::default::Default::default();
                            if prepared_precision.is_some() {
                                table_writer.write_entry::<self::Precision>(0);
                            }

                            unsafe {
                                table_writer.finish(builder, |object_writer| {
                                    if let ::core::option::Option::Some(prepared_precision) =
                                        prepared_precision
                                    {
                                        object_writer.write::<_, _, 2>(&prepared_precision);
                                    }
                                });
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<FloatingPoint>> for FloatingPoint {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<FloatingPoint> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<FloatingPoint>> for FloatingPoint {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<FloatingPoint>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<FloatingPoint> for FloatingPoint {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<FloatingPoint> {
                            FloatingPoint::create(builder, self.precision)
                        }
                    }

                    /// Builder for serializing an instance of the [FloatingPoint] type.
                    ///
                    /// Can be created using the [FloatingPoint::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct FloatingPointBuilder<State>(State);

                    impl FloatingPointBuilder<()> {
                        /// Setter for the [`precision` field](FloatingPoint#structfield.precision).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn precision<T0>(self, value: T0) -> FloatingPointBuilder<(T0,)>
                        where
                            T0: ::planus::WriteAsDefault<self::Precision, self::Precision>,
                        {
                            FloatingPointBuilder((value,))
                        }

                        /// Sets the [`precision` field](FloatingPoint#structfield.precision) to the default value.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn precision_as_default(
                            self,
                        ) -> FloatingPointBuilder<(::planus::DefaultValue,)>
                        {
                            self.precision(::planus::DefaultValue)
                        }
                    }

                    impl<T0> FloatingPointBuilder<(T0,)> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [FloatingPoint].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<FloatingPoint>
                        where
                            Self: ::planus::WriteAsOffset<FloatingPoint>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl<T0: ::planus::WriteAsDefault<self::Precision, self::Precision>>
                        ::planus::WriteAs<::planus::Offset<FloatingPoint>>
                        for FloatingPointBuilder<(T0,)>
                    {
                        type Prepared = ::planus::Offset<FloatingPoint>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<FloatingPoint> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl<T0: ::planus::WriteAsDefault<self::Precision, self::Precision>>
                        ::planus::WriteAsOptional<::planus::Offset<FloatingPoint>>
                        for FloatingPointBuilder<(T0,)>
                    {
                        type Prepared = ::planus::Offset<FloatingPoint>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<FloatingPoint>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl<T0: ::planus::WriteAsDefault<self::Precision, self::Precision>>
                        ::planus::WriteAsOffset<FloatingPoint> for FloatingPointBuilder<(T0,)>
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<FloatingPoint> {
                            let (v0,) = &self.0;
                            FloatingPoint::create(builder, v0)
                        }
                    }

                    /// Reference to a deserialized [FloatingPoint].
                    #[derive(Copy, Clone)]
                    pub struct FloatingPointRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> FloatingPointRef<'a> {
                        /// Getter for the [`precision` field](FloatingPoint#structfield.precision).
                        #[inline]
                        pub fn precision(&self) -> ::planus::Result<self::Precision> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(0, "FloatingPoint", "precision")?
                                    .unwrap_or(self::Precision::Half),
                            )
                        }
                    }

                    impl<'a> ::core::fmt::Debug for FloatingPointRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("FloatingPointRef");
                            f.field("precision", &self.precision());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<FloatingPointRef<'a>> for FloatingPoint {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: FloatingPointRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                precision: ::core::convert::TryInto::try_into(value.precision()?)?,
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for FloatingPointRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for FloatingPointRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[FloatingPointRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<FloatingPoint>> for FloatingPoint {
                        type Value = ::planus::Offset<FloatingPoint>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<FloatingPoint>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for FloatingPointRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[FloatingPointRef]",
                                    "read_as_root",
                                    0,
                                )
                            })
                        }
                    }

                    ///  Unicode with UTF-8 encoding
                    ///
                    /// Generated from these locations:
                    /// * Table `Utf8` in the file `Schema.fbs:171`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct Utf8 {}

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for Utf8 {
                        fn default() -> Self {
                            Self {}
                        }
                    }

                    impl Utf8 {
                        /// Creates a [Utf8Builder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> Utf8Builder<()> {
                            Utf8Builder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(builder: &mut ::planus::Builder) -> ::planus::Offset<Self> {
                            let table_writer: ::planus::table_writer::TableWriter<4> =
                                ::core::default::Default::default();
                            unsafe {
                                table_writer.finish(builder, |_table_writer| {});
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Utf8>> for Utf8 {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Utf8> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Utf8>> for Utf8 {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Utf8>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Utf8> for Utf8 {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Utf8> {
                            Utf8::create(builder)
                        }
                    }

                    /// Builder for serializing an instance of the [Utf8] type.
                    ///
                    /// Can be created using the [Utf8::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct Utf8Builder<State>(State);

                    impl Utf8Builder<()> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Utf8].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Utf8>
                        where
                            Self: ::planus::WriteAsOffset<Utf8>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Utf8>> for Utf8Builder<()> {
                        type Prepared = ::planus::Offset<Utf8>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Utf8> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Utf8>> for Utf8Builder<()> {
                        type Prepared = ::planus::Offset<Utf8>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Utf8>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Utf8> for Utf8Builder<()> {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Utf8> {
                            Utf8::create(builder)
                        }
                    }

                    /// Reference to a deserialized [Utf8].
                    #[derive(Copy, Clone)]
                    pub struct Utf8Ref<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> Utf8Ref<'a> {}

                    impl<'a> ::core::fmt::Debug for Utf8Ref<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("Utf8Ref");

                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<Utf8Ref<'a>> for Utf8 {
                        type Error = ::planus::Error;

                        fn try_from(_value: Utf8Ref<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {})
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for Utf8Ref<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for Utf8Ref<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[Utf8Ref]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<Utf8>> for Utf8 {
                        type Value = ::planus::Offset<Utf8>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Utf8>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for Utf8Ref<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[Utf8Ref]", "read_as_root", 0)
                            })
                        }
                    }

                    ///  Opaque binary data
                    ///
                    /// Generated from these locations:
                    /// * Table `Binary` in the file `Schema.fbs:175`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct Binary {}

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for Binary {
                        fn default() -> Self {
                            Self {}
                        }
                    }

                    impl Binary {
                        /// Creates a [BinaryBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> BinaryBuilder<()> {
                            BinaryBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(builder: &mut ::planus::Builder) -> ::planus::Offset<Self> {
                            let table_writer: ::planus::table_writer::TableWriter<4> =
                                ::core::default::Default::default();
                            unsafe {
                                table_writer.finish(builder, |_table_writer| {});
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Binary>> for Binary {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Binary> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Binary>> for Binary {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Binary>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Binary> for Binary {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Binary> {
                            Binary::create(builder)
                        }
                    }

                    /// Builder for serializing an instance of the [Binary] type.
                    ///
                    /// Can be created using the [Binary::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct BinaryBuilder<State>(State);

                    impl BinaryBuilder<()> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Binary].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Binary>
                        where
                            Self: ::planus::WriteAsOffset<Binary>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Binary>> for BinaryBuilder<()> {
                        type Prepared = ::planus::Offset<Binary>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Binary> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Binary>> for BinaryBuilder<()> {
                        type Prepared = ::planus::Offset<Binary>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Binary>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Binary> for BinaryBuilder<()> {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Binary> {
                            Binary::create(builder)
                        }
                    }

                    /// Reference to a deserialized [Binary].
                    #[derive(Copy, Clone)]
                    pub struct BinaryRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> BinaryRef<'a> {}

                    impl<'a> ::core::fmt::Debug for BinaryRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("BinaryRef");

                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<BinaryRef<'a>> for Binary {
                        type Error = ::planus::Error;

                        fn try_from(_value: BinaryRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {})
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for BinaryRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for BinaryRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[BinaryRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<Binary>> for Binary {
                        type Value = ::planus::Offset<Binary>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Binary>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for BinaryRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[BinaryRef]", "read_as_root", 0)
                            })
                        }
                    }

                    ///  Same as Utf8, but with 64-bit offsets, allowing to represent
                    ///  extremely large data values.
                    ///
                    /// Generated from these locations:
                    /// * Table `LargeUtf8` in the file `Schema.fbs:180`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct LargeUtf8 {}

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for LargeUtf8 {
                        fn default() -> Self {
                            Self {}
                        }
                    }

                    impl LargeUtf8 {
                        /// Creates a [LargeUtf8Builder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> LargeUtf8Builder<()> {
                            LargeUtf8Builder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(builder: &mut ::planus::Builder) -> ::planus::Offset<Self> {
                            let table_writer: ::planus::table_writer::TableWriter<4> =
                                ::core::default::Default::default();
                            unsafe {
                                table_writer.finish(builder, |_table_writer| {});
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<LargeUtf8>> for LargeUtf8 {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<LargeUtf8> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<LargeUtf8>> for LargeUtf8 {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<LargeUtf8>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<LargeUtf8> for LargeUtf8 {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<LargeUtf8> {
                            LargeUtf8::create(builder)
                        }
                    }

                    /// Builder for serializing an instance of the [LargeUtf8] type.
                    ///
                    /// Can be created using the [LargeUtf8::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct LargeUtf8Builder<State>(State);

                    impl LargeUtf8Builder<()> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [LargeUtf8].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<LargeUtf8>
                        where
                            Self: ::planus::WriteAsOffset<LargeUtf8>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<LargeUtf8>> for LargeUtf8Builder<()> {
                        type Prepared = ::planus::Offset<LargeUtf8>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<LargeUtf8> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<LargeUtf8>> for LargeUtf8Builder<()> {
                        type Prepared = ::planus::Offset<LargeUtf8>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<LargeUtf8>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<LargeUtf8> for LargeUtf8Builder<()> {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<LargeUtf8> {
                            LargeUtf8::create(builder)
                        }
                    }

                    /// Reference to a deserialized [LargeUtf8].
                    #[derive(Copy, Clone)]
                    pub struct LargeUtf8Ref<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> LargeUtf8Ref<'a> {}

                    impl<'a> ::core::fmt::Debug for LargeUtf8Ref<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("LargeUtf8Ref");

                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<LargeUtf8Ref<'a>> for LargeUtf8 {
                        type Error = ::planus::Error;

                        fn try_from(_value: LargeUtf8Ref<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {})
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for LargeUtf8Ref<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for LargeUtf8Ref<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[LargeUtf8Ref]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<LargeUtf8>> for LargeUtf8 {
                        type Value = ::planus::Offset<LargeUtf8>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<LargeUtf8>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for LargeUtf8Ref<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[LargeUtf8Ref]", "read_as_root", 0)
                            })
                        }
                    }

                    ///  Same as Binary, but with 64-bit offsets, allowing to represent
                    ///  extremely large data values.
                    ///
                    /// Generated from these locations:
                    /// * Table `LargeBinary` in the file `Schema.fbs:185`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct LargeBinary {}

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for LargeBinary {
                        fn default() -> Self {
                            Self {}
                        }
                    }

                    impl LargeBinary {
                        /// Creates a [LargeBinaryBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> LargeBinaryBuilder<()> {
                            LargeBinaryBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(builder: &mut ::planus::Builder) -> ::planus::Offset<Self> {
                            let table_writer: ::planus::table_writer::TableWriter<4> =
                                ::core::default::Default::default();
                            unsafe {
                                table_writer.finish(builder, |_table_writer| {});
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<LargeBinary>> for LargeBinary {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<LargeBinary> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<LargeBinary>> for LargeBinary {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<LargeBinary>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<LargeBinary> for LargeBinary {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<LargeBinary> {
                            LargeBinary::create(builder)
                        }
                    }

                    /// Builder for serializing an instance of the [LargeBinary] type.
                    ///
                    /// Can be created using the [LargeBinary::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct LargeBinaryBuilder<State>(State);

                    impl LargeBinaryBuilder<()> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [LargeBinary].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<LargeBinary>
                        where
                            Self: ::planus::WriteAsOffset<LargeBinary>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<LargeBinary>> for LargeBinaryBuilder<()> {
                        type Prepared = ::planus::Offset<LargeBinary>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<LargeBinary> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<LargeBinary>> for LargeBinaryBuilder<()> {
                        type Prepared = ::planus::Offset<LargeBinary>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<LargeBinary>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<LargeBinary> for LargeBinaryBuilder<()> {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<LargeBinary> {
                            LargeBinary::create(builder)
                        }
                    }

                    /// Reference to a deserialized [LargeBinary].
                    #[derive(Copy, Clone)]
                    pub struct LargeBinaryRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> LargeBinaryRef<'a> {}

                    impl<'a> ::core::fmt::Debug for LargeBinaryRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("LargeBinaryRef");

                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<LargeBinaryRef<'a>> for LargeBinary {
                        type Error = ::planus::Error;

                        fn try_from(_value: LargeBinaryRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {})
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for LargeBinaryRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for LargeBinaryRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[LargeBinaryRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<LargeBinary>> for LargeBinary {
                        type Value = ::planus::Offset<LargeBinary>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<LargeBinary>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for LargeBinaryRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[LargeBinaryRef]",
                                    "read_as_root",
                                    0,
                                )
                            })
                        }
                    }

                    ///  Logically the same as Utf8, but the internal representation uses a view
                    ///  struct that contains the string length and either the string's entire data
                    ///  inline (for small strings) or an inlined prefix, an index of another buffer,
                    ///  and an offset pointing to a slice in that buffer (for non-small strings).
                    ///
                    ///  Since it uses a variable number of data buffers, each Field with this type
                    ///  must have a corresponding entry in `variadicBufferCounts`.
                    ///
                    /// Generated from these locations:
                    /// * Table `Utf8View` in the file `Schema.fbs:195`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct Utf8View {}

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for Utf8View {
                        fn default() -> Self {
                            Self {}
                        }
                    }

                    impl Utf8View {
                        /// Creates a [Utf8ViewBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> Utf8ViewBuilder<()> {
                            Utf8ViewBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(builder: &mut ::planus::Builder) -> ::planus::Offset<Self> {
                            let table_writer: ::planus::table_writer::TableWriter<4> =
                                ::core::default::Default::default();
                            unsafe {
                                table_writer.finish(builder, |_table_writer| {});
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Utf8View>> for Utf8View {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Utf8View> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Utf8View>> for Utf8View {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Utf8View>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Utf8View> for Utf8View {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Utf8View> {
                            Utf8View::create(builder)
                        }
                    }

                    /// Builder for serializing an instance of the [Utf8View] type.
                    ///
                    /// Can be created using the [Utf8View::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct Utf8ViewBuilder<State>(State);

                    impl Utf8ViewBuilder<()> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Utf8View].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Utf8View>
                        where
                            Self: ::planus::WriteAsOffset<Utf8View>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Utf8View>> for Utf8ViewBuilder<()> {
                        type Prepared = ::planus::Offset<Utf8View>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Utf8View> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Utf8View>> for Utf8ViewBuilder<()> {
                        type Prepared = ::planus::Offset<Utf8View>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Utf8View>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Utf8View> for Utf8ViewBuilder<()> {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Utf8View> {
                            Utf8View::create(builder)
                        }
                    }

                    /// Reference to a deserialized [Utf8View].
                    #[derive(Copy, Clone)]
                    pub struct Utf8ViewRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> Utf8ViewRef<'a> {}

                    impl<'a> ::core::fmt::Debug for Utf8ViewRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("Utf8ViewRef");

                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<Utf8ViewRef<'a>> for Utf8View {
                        type Error = ::planus::Error;

                        fn try_from(_value: Utf8ViewRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {})
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for Utf8ViewRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for Utf8ViewRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[Utf8ViewRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<Utf8View>> for Utf8View {
                        type Value = ::planus::Offset<Utf8View>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Utf8View>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for Utf8ViewRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[Utf8ViewRef]", "read_as_root", 0)
                            })
                        }
                    }

                    ///  Logically the same as Binary, but the internal representation uses a view
                    ///  struct that contains the string length and either the string's entire data
                    ///  inline (for small strings) or an inlined prefix, an index of another buffer,
                    ///  and an offset pointing to a slice in that buffer (for non-small strings).
                    ///
                    ///  Since it uses a variable number of data buffers, each Field with this type
                    ///  must have a corresponding entry in `variadicBufferCounts`.
                    ///
                    /// Generated from these locations:
                    /// * Table `BinaryView` in the file `Schema.fbs:205`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct BinaryView {}

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for BinaryView {
                        fn default() -> Self {
                            Self {}
                        }
                    }

                    impl BinaryView {
                        /// Creates a [BinaryViewBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> BinaryViewBuilder<()> {
                            BinaryViewBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(builder: &mut ::planus::Builder) -> ::planus::Offset<Self> {
                            let table_writer: ::planus::table_writer::TableWriter<4> =
                                ::core::default::Default::default();
                            unsafe {
                                table_writer.finish(builder, |_table_writer| {});
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<BinaryView>> for BinaryView {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<BinaryView> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<BinaryView>> for BinaryView {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<BinaryView>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<BinaryView> for BinaryView {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<BinaryView> {
                            BinaryView::create(builder)
                        }
                    }

                    /// Builder for serializing an instance of the [BinaryView] type.
                    ///
                    /// Can be created using the [BinaryView::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct BinaryViewBuilder<State>(State);

                    impl BinaryViewBuilder<()> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [BinaryView].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<BinaryView>
                        where
                            Self: ::planus::WriteAsOffset<BinaryView>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<BinaryView>> for BinaryViewBuilder<()> {
                        type Prepared = ::planus::Offset<BinaryView>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<BinaryView> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<BinaryView>> for BinaryViewBuilder<()> {
                        type Prepared = ::planus::Offset<BinaryView>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<BinaryView>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<BinaryView> for BinaryViewBuilder<()> {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<BinaryView> {
                            BinaryView::create(builder)
                        }
                    }

                    /// Reference to a deserialized [BinaryView].
                    #[derive(Copy, Clone)]
                    pub struct BinaryViewRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> BinaryViewRef<'a> {}

                    impl<'a> ::core::fmt::Debug for BinaryViewRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("BinaryViewRef");

                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<BinaryViewRef<'a>> for BinaryView {
                        type Error = ::planus::Error;

                        fn try_from(_value: BinaryViewRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {})
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for BinaryViewRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for BinaryViewRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[BinaryViewRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<BinaryView>> for BinaryView {
                        type Value = ::planus::Offset<BinaryView>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<BinaryView>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for BinaryViewRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[BinaryViewRef]", "read_as_root", 0)
                            })
                        }
                    }

                    /// The table `FixedSizeBinary` in the namespace `org.apache.arrow.flatbuf`
                    ///
                    /// Generated from these locations:
                    /// * Table `FixedSizeBinary` in the file `Schema.fbs:209`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct FixedSizeBinary {
                        ///  Number of bytes per value
                        pub byte_width: i32,
                    }

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for FixedSizeBinary {
                        fn default() -> Self {
                            Self { byte_width: 0 }
                        }
                    }

                    impl FixedSizeBinary {
                        /// Creates a [FixedSizeBinaryBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> FixedSizeBinaryBuilder<()> {
                            FixedSizeBinaryBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            field_byte_width: impl ::planus::WriteAsDefault<i32, i32>,
                        ) -> ::planus::Offset<Self> {
                            let prepared_byte_width = field_byte_width.prepare(builder, &0);

                            let mut table_writer: ::planus::table_writer::TableWriter<6> =
                                ::core::default::Default::default();
                            if prepared_byte_width.is_some() {
                                table_writer.write_entry::<i32>(0);
                            }

                            unsafe {
                                table_writer.finish(builder, |object_writer| {
                                    if let ::core::option::Option::Some(prepared_byte_width) =
                                        prepared_byte_width
                                    {
                                        object_writer.write::<_, _, 4>(&prepared_byte_width);
                                    }
                                });
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<FixedSizeBinary>> for FixedSizeBinary {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<FixedSizeBinary> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<FixedSizeBinary>> for FixedSizeBinary {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<FixedSizeBinary>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<FixedSizeBinary> for FixedSizeBinary {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<FixedSizeBinary> {
                            FixedSizeBinary::create(builder, self.byte_width)
                        }
                    }

                    /// Builder for serializing an instance of the [FixedSizeBinary] type.
                    ///
                    /// Can be created using the [FixedSizeBinary::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct FixedSizeBinaryBuilder<State>(State);

                    impl FixedSizeBinaryBuilder<()> {
                        /// Setter for the [`byteWidth` field](FixedSizeBinary#structfield.byte_width).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn byte_width<T0>(self, value: T0) -> FixedSizeBinaryBuilder<(T0,)>
                        where
                            T0: ::planus::WriteAsDefault<i32, i32>,
                        {
                            FixedSizeBinaryBuilder((value,))
                        }

                        /// Sets the [`byteWidth` field](FixedSizeBinary#structfield.byte_width) to the default value.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn byte_width_as_default(
                            self,
                        ) -> FixedSizeBinaryBuilder<(::planus::DefaultValue,)>
                        {
                            self.byte_width(::planus::DefaultValue)
                        }
                    }

                    impl<T0> FixedSizeBinaryBuilder<(T0,)> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [FixedSizeBinary].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<FixedSizeBinary>
                        where
                            Self: ::planus::WriteAsOffset<FixedSizeBinary>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl<T0: ::planus::WriteAsDefault<i32, i32>>
                        ::planus::WriteAs<::planus::Offset<FixedSizeBinary>>
                        for FixedSizeBinaryBuilder<(T0,)>
                    {
                        type Prepared = ::planus::Offset<FixedSizeBinary>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<FixedSizeBinary> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl<T0: ::planus::WriteAsDefault<i32, i32>>
                        ::planus::WriteAsOptional<::planus::Offset<FixedSizeBinary>>
                        for FixedSizeBinaryBuilder<(T0,)>
                    {
                        type Prepared = ::planus::Offset<FixedSizeBinary>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<FixedSizeBinary>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl<T0: ::planus::WriteAsDefault<i32, i32>>
                        ::planus::WriteAsOffset<FixedSizeBinary> for FixedSizeBinaryBuilder<(T0,)>
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<FixedSizeBinary> {
                            let (v0,) = &self.0;
                            FixedSizeBinary::create(builder, v0)
                        }
                    }

                    /// Reference to a deserialized [FixedSizeBinary].
                    #[derive(Copy, Clone)]
                    pub struct FixedSizeBinaryRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> FixedSizeBinaryRef<'a> {
                        /// Getter for the [`byteWidth` field](FixedSizeBinary#structfield.byte_width).
                        #[inline]
                        pub fn byte_width(&self) -> ::planus::Result<i32> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(0, "FixedSizeBinary", "byte_width")?
                                    .unwrap_or(0),
                            )
                        }
                    }

                    impl<'a> ::core::fmt::Debug for FixedSizeBinaryRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("FixedSizeBinaryRef");
                            f.field("byte_width", &self.byte_width());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<FixedSizeBinaryRef<'a>> for FixedSizeBinary {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: FixedSizeBinaryRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                byte_width: ::core::convert::TryInto::try_into(
                                    value.byte_width()?,
                                )?,
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for FixedSizeBinaryRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for FixedSizeBinaryRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[FixedSizeBinaryRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<FixedSizeBinary>> for FixedSizeBinary {
                        type Value = ::planus::Offset<FixedSizeBinary>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<FixedSizeBinary>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for FixedSizeBinaryRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[FixedSizeBinaryRef]",
                                    "read_as_root",
                                    0,
                                )
                            })
                        }
                    }

                    /// The table `Bool` in the namespace `org.apache.arrow.flatbuf`
                    ///
                    /// Generated from these locations:
                    /// * Table `Bool` in the file `Schema.fbs:214`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct Bool {}

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for Bool {
                        fn default() -> Self {
                            Self {}
                        }
                    }

                    impl Bool {
                        /// Creates a [BoolBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> BoolBuilder<()> {
                            BoolBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(builder: &mut ::planus::Builder) -> ::planus::Offset<Self> {
                            let table_writer: ::planus::table_writer::TableWriter<4> =
                                ::core::default::Default::default();
                            unsafe {
                                table_writer.finish(builder, |_table_writer| {});
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Bool>> for Bool {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Bool> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Bool>> for Bool {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Bool>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Bool> for Bool {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Bool> {
                            Bool::create(builder)
                        }
                    }

                    /// Builder for serializing an instance of the [Bool] type.
                    ///
                    /// Can be created using the [Bool::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct BoolBuilder<State>(State);

                    impl BoolBuilder<()> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Bool].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Bool>
                        where
                            Self: ::planus::WriteAsOffset<Bool>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Bool>> for BoolBuilder<()> {
                        type Prepared = ::planus::Offset<Bool>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Bool> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Bool>> for BoolBuilder<()> {
                        type Prepared = ::planus::Offset<Bool>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Bool>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Bool> for BoolBuilder<()> {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Bool> {
                            Bool::create(builder)
                        }
                    }

                    /// Reference to a deserialized [Bool].
                    #[derive(Copy, Clone)]
                    pub struct BoolRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> BoolRef<'a> {}

                    impl<'a> ::core::fmt::Debug for BoolRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("BoolRef");

                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<BoolRef<'a>> for Bool {
                        type Error = ::planus::Error;

                        fn try_from(_value: BoolRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {})
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for BoolRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for BoolRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[BoolRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<Bool>> for Bool {
                        type Value = ::planus::Offset<Bool>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Bool>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for BoolRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[BoolRef]", "read_as_root", 0)
                            })
                        }
                    }

                    ///  Contains two child arrays, run_ends and values.
                    ///  The run_ends child array must be a 16/32/64-bit integer array
                    ///  which encodes the indices at which the run with the value in
                    ///  each corresponding index in the values child array ends.
                    ///  Like list/struct types, the value array can be of any type.
                    ///
                    /// Generated from these locations:
                    /// * Table `RunEndEncoded` in the file `Schema.fbs:222`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct RunEndEncoded {}

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for RunEndEncoded {
                        fn default() -> Self {
                            Self {}
                        }
                    }

                    impl RunEndEncoded {
                        /// Creates a [RunEndEncodedBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> RunEndEncodedBuilder<()> {
                            RunEndEncodedBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(builder: &mut ::planus::Builder) -> ::planus::Offset<Self> {
                            let table_writer: ::planus::table_writer::TableWriter<4> =
                                ::core::default::Default::default();
                            unsafe {
                                table_writer.finish(builder, |_table_writer| {});
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<RunEndEncoded>> for RunEndEncoded {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<RunEndEncoded> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<RunEndEncoded>> for RunEndEncoded {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<RunEndEncoded>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<RunEndEncoded> for RunEndEncoded {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<RunEndEncoded> {
                            RunEndEncoded::create(builder)
                        }
                    }

                    /// Builder for serializing an instance of the [RunEndEncoded] type.
                    ///
                    /// Can be created using the [RunEndEncoded::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct RunEndEncodedBuilder<State>(State);

                    impl RunEndEncodedBuilder<()> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [RunEndEncoded].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<RunEndEncoded>
                        where
                            Self: ::planus::WriteAsOffset<RunEndEncoded>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<RunEndEncoded>> for RunEndEncodedBuilder<()> {
                        type Prepared = ::planus::Offset<RunEndEncoded>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<RunEndEncoded> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<RunEndEncoded>> for RunEndEncodedBuilder<()> {
                        type Prepared = ::planus::Offset<RunEndEncoded>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<RunEndEncoded>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<RunEndEncoded> for RunEndEncodedBuilder<()> {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<RunEndEncoded> {
                            RunEndEncoded::create(builder)
                        }
                    }

                    /// Reference to a deserialized [RunEndEncoded].
                    #[derive(Copy, Clone)]
                    pub struct RunEndEncodedRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> RunEndEncodedRef<'a> {}

                    impl<'a> ::core::fmt::Debug for RunEndEncodedRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("RunEndEncodedRef");

                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<RunEndEncodedRef<'a>> for RunEndEncoded {
                        type Error = ::planus::Error;

                        fn try_from(_value: RunEndEncodedRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {})
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for RunEndEncodedRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for RunEndEncodedRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[RunEndEncodedRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<RunEndEncoded>> for RunEndEncoded {
                        type Value = ::planus::Offset<RunEndEncoded>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<RunEndEncoded>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for RunEndEncodedRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[RunEndEncodedRef]",
                                    "read_as_root",
                                    0,
                                )
                            })
                        }
                    }

                    ///  Exact decimal value represented as an integer value in two's
                    ///  complement. Currently 32-bit (4-byte), 64-bit (8-byte),
                    ///  128-bit (16-byte) and 256-bit (32-byte) integers are used.
                    ///  The representation uses the endianness indicated in the Schema.
                    ///
                    /// Generated from these locations:
                    /// * Table `Decimal` in the file `Schema.fbs:229`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct Decimal {
                        ///  Total number of decimal digits
                        pub precision: i32,
                        ///  Number of digits after the decimal point "."
                        pub scale: i32,
                        ///  Number of bits per value. The accepted widths are 32, 64, 128 and 256.
                        ///  We use bitWidth for consistency with Int::bitWidth.
                        pub bit_width: i32,
                    }

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for Decimal {
                        fn default() -> Self {
                            Self {
                                precision: 0,
                                scale: 0,
                                bit_width: 128,
                            }
                        }
                    }

                    impl Decimal {
                        /// Creates a [DecimalBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> DecimalBuilder<()> {
                            DecimalBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            field_precision: impl ::planus::WriteAsDefault<i32, i32>,
                            field_scale: impl ::planus::WriteAsDefault<i32, i32>,
                            field_bit_width: impl ::planus::WriteAsDefault<i32, i32>,
                        ) -> ::planus::Offset<Self> {
                            let prepared_precision = field_precision.prepare(builder, &0);
                            let prepared_scale = field_scale.prepare(builder, &0);
                            let prepared_bit_width = field_bit_width.prepare(builder, &128);

                            let mut table_writer: ::planus::table_writer::TableWriter<10> =
                                ::core::default::Default::default();
                            if prepared_precision.is_some() {
                                table_writer.write_entry::<i32>(0);
                            }
                            if prepared_scale.is_some() {
                                table_writer.write_entry::<i32>(1);
                            }
                            if prepared_bit_width.is_some() {
                                table_writer.write_entry::<i32>(2);
                            }

                            unsafe {
                                table_writer.finish(builder, |object_writer| {
                                    if let ::core::option::Option::Some(prepared_precision) =
                                        prepared_precision
                                    {
                                        object_writer.write::<_, _, 4>(&prepared_precision);
                                    }
                                    if let ::core::option::Option::Some(prepared_scale) =
                                        prepared_scale
                                    {
                                        object_writer.write::<_, _, 4>(&prepared_scale);
                                    }
                                    if let ::core::option::Option::Some(prepared_bit_width) =
                                        prepared_bit_width
                                    {
                                        object_writer.write::<_, _, 4>(&prepared_bit_width);
                                    }
                                });
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Decimal>> for Decimal {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Decimal> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Decimal>> for Decimal {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Decimal>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Decimal> for Decimal {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Decimal> {
                            Decimal::create(builder, self.precision, self.scale, self.bit_width)
                        }
                    }

                    /// Builder for serializing an instance of the [Decimal] type.
                    ///
                    /// Can be created using the [Decimal::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct DecimalBuilder<State>(State);

                    impl DecimalBuilder<()> {
                        /// Setter for the [`precision` field](Decimal#structfield.precision).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn precision<T0>(self, value: T0) -> DecimalBuilder<(T0,)>
                        where
                            T0: ::planus::WriteAsDefault<i32, i32>,
                        {
                            DecimalBuilder((value,))
                        }

                        /// Sets the [`precision` field](Decimal#structfield.precision) to the default value.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn precision_as_default(
                            self,
                        ) -> DecimalBuilder<(::planus::DefaultValue,)> {
                            self.precision(::planus::DefaultValue)
                        }
                    }

                    impl<T0> DecimalBuilder<(T0,)> {
                        /// Setter for the [`scale` field](Decimal#structfield.scale).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn scale<T1>(self, value: T1) -> DecimalBuilder<(T0, T1)>
                        where
                            T1: ::planus::WriteAsDefault<i32, i32>,
                        {
                            let (v0,) = self.0;
                            DecimalBuilder((v0, value))
                        }

                        /// Sets the [`scale` field](Decimal#structfield.scale) to the default value.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn scale_as_default(
                            self,
                        ) -> DecimalBuilder<(T0, ::planus::DefaultValue)> {
                            self.scale(::planus::DefaultValue)
                        }
                    }

                    impl<T0, T1> DecimalBuilder<(T0, T1)> {
                        /// Setter for the [`bitWidth` field](Decimal#structfield.bit_width).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn bit_width<T2>(self, value: T2) -> DecimalBuilder<(T0, T1, T2)>
                        where
                            T2: ::planus::WriteAsDefault<i32, i32>,
                        {
                            let (v0, v1) = self.0;
                            DecimalBuilder((v0, v1, value))
                        }

                        /// Sets the [`bitWidth` field](Decimal#structfield.bit_width) to the default value.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn bit_width_as_default(
                            self,
                        ) -> DecimalBuilder<(T0, T1, ::planus::DefaultValue)>
                        {
                            self.bit_width(::planus::DefaultValue)
                        }
                    }

                    impl<T0, T1, T2> DecimalBuilder<(T0, T1, T2)> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Decimal].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Decimal>
                        where
                            Self: ::planus::WriteAsOffset<Decimal>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<i32, i32>,
                            T1: ::planus::WriteAsDefault<i32, i32>,
                            T2: ::planus::WriteAsDefault<i32, i32>,
                        > ::planus::WriteAs<::planus::Offset<Decimal>>
                        for DecimalBuilder<(T0, T1, T2)>
                    {
                        type Prepared = ::planus::Offset<Decimal>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Decimal> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<i32, i32>,
                            T1: ::planus::WriteAsDefault<i32, i32>,
                            T2: ::planus::WriteAsDefault<i32, i32>,
                        > ::planus::WriteAsOptional<::planus::Offset<Decimal>>
                        for DecimalBuilder<(T0, T1, T2)>
                    {
                        type Prepared = ::planus::Offset<Decimal>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Decimal>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<i32, i32>,
                            T1: ::planus::WriteAsDefault<i32, i32>,
                            T2: ::planus::WriteAsDefault<i32, i32>,
                        > ::planus::WriteAsOffset<Decimal> for DecimalBuilder<(T0, T1, T2)>
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Decimal> {
                            let (v0, v1, v2) = &self.0;
                            Decimal::create(builder, v0, v1, v2)
                        }
                    }

                    /// Reference to a deserialized [Decimal].
                    #[derive(Copy, Clone)]
                    pub struct DecimalRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> DecimalRef<'a> {
                        /// Getter for the [`precision` field](Decimal#structfield.precision).
                        #[inline]
                        pub fn precision(&self) -> ::planus::Result<i32> {
                            ::core::result::Result::Ok(
                                self.0.access(0, "Decimal", "precision")?.unwrap_or(0),
                            )
                        }

                        /// Getter for the [`scale` field](Decimal#structfield.scale).
                        #[inline]
                        pub fn scale(&self) -> ::planus::Result<i32> {
                            ::core::result::Result::Ok(
                                self.0.access(1, "Decimal", "scale")?.unwrap_or(0),
                            )
                        }

                        /// Getter for the [`bitWidth` field](Decimal#structfield.bit_width).
                        #[inline]
                        pub fn bit_width(&self) -> ::planus::Result<i32> {
                            ::core::result::Result::Ok(
                                self.0.access(2, "Decimal", "bit_width")?.unwrap_or(128),
                            )
                        }
                    }

                    impl<'a> ::core::fmt::Debug for DecimalRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("DecimalRef");
                            f.field("precision", &self.precision());
                            f.field("scale", &self.scale());
                            f.field("bit_width", &self.bit_width());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<DecimalRef<'a>> for Decimal {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: DecimalRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                precision: ::core::convert::TryInto::try_into(value.precision()?)?,
                                scale: ::core::convert::TryInto::try_into(value.scale()?)?,
                                bit_width: ::core::convert::TryInto::try_into(value.bit_width()?)?,
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for DecimalRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for DecimalRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[DecimalRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<Decimal>> for Decimal {
                        type Value = ::planus::Offset<Decimal>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Decimal>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for DecimalRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[DecimalRef]", "read_as_root", 0)
                            })
                        }
                    }

                    /// The enum `DateUnit` in the namespace `org.apache.arrow.flatbuf`
                    ///
                    /// Generated from these locations:
                    /// * Enum `DateUnit` in the file `Schema.fbs:241`
                    #[derive(
                        Copy,
                        Clone,
                        Debug,
                        PartialEq,
                        Eq,
                        PartialOrd,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    #[repr(i16)]
                    pub enum DateUnit {
                        /// The variant `DAY` in the enum `DateUnit`
                        Day = 0,

                        /// The variant `MILLISECOND` in the enum `DateUnit`
                        Millisecond = 1,
                    }

                    impl DateUnit {
                        /// Array containing all valid variants of DateUnit
                        pub const ENUM_VALUES: [Self; 2] = [Self::Day, Self::Millisecond];
                    }

                    impl ::core::convert::TryFrom<i16> for DateUnit {
                        type Error = ::planus::errors::UnknownEnumTagKind;
                        #[inline]
                        fn try_from(
                            value: i16,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTagKind>
                        {
                            #[allow(clippy::match_single_binding)]
                            match value {
                                0 => ::core::result::Result::Ok(DateUnit::Day),
                                1 => ::core::result::Result::Ok(DateUnit::Millisecond),

                                _ => ::core::result::Result::Err(
                                    ::planus::errors::UnknownEnumTagKind { tag: value as i128 },
                                ),
                            }
                        }
                    }

                    impl ::core::convert::From<DateUnit> for i16 {
                        #[inline]
                        fn from(value: DateUnit) -> Self {
                            value as i16
                        }
                    }

                    /// # Safety
                    /// The Planus compiler correctly calculates `ALIGNMENT` and `SIZE`.
                    unsafe impl ::planus::Primitive for DateUnit {
                        const ALIGNMENT: usize = 2;
                        const SIZE: usize = 2;
                    }

                    impl ::planus::WriteAsPrimitive<DateUnit> for DateUnit {
                        #[inline]
                        fn write<const N: usize>(
                            &self,
                            cursor: ::planus::Cursor<'_, N>,
                            buffer_position: u32,
                        ) {
                            (*self as i16).write(cursor, buffer_position);
                        }
                    }

                    impl ::planus::WriteAs<DateUnit> for DateUnit {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> DateUnit {
                            *self
                        }
                    }

                    impl ::planus::WriteAsDefault<DateUnit, DateUnit> for DateUnit {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                            default: &DateUnit,
                        ) -> ::core::option::Option<DateUnit> {
                            if self == default {
                                ::core::option::Option::None
                            } else {
                                ::core::option::Option::Some(*self)
                            }
                        }
                    }

                    impl ::planus::WriteAsOptional<DateUnit> for DateUnit {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<DateUnit> {
                            ::core::option::Option::Some(*self)
                        }
                    }

                    impl<'buf> ::planus::TableRead<'buf> for DateUnit {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            let n: i16 = ::planus::TableRead::from_buffer(buffer, offset)?;
                            ::core::result::Result::Ok(::core::convert::TryInto::try_into(n)?)
                        }
                    }

                    impl<'buf> ::planus::VectorReadInner<'buf> for DateUnit {
                        type Error = ::planus::errors::UnknownEnumTag;
                        const STRIDE: usize = 2;
                        #[inline]
                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTag>
                        {
                            let value = unsafe {
                                <i16 as ::planus::VectorRead>::from_buffer(buffer, offset)
                            };
                            let value: ::core::result::Result<Self, _> =
                                ::core::convert::TryInto::try_into(value);
                            value.map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "DateUnit",
                                    "VectorRead::from_buffer",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<DateUnit> for DateUnit {
                        const STRIDE: usize = 2;

                        type Value = Self;

                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self {
                            *self
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[Self],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 2];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (2 * i) as u32,
                                );
                            }
                        }
                    }

                    ///  Date is either a 32-bit or 64-bit signed integer type representing an
                    ///  elapsed time since UNIX epoch (1970-01-01), stored in either of two units:
                    ///
                    ///  * Milliseconds (64 bits) indicating UNIX time elapsed since the epoch (no
                    ///    leap seconds), where the values are evenly divisible by 86400000
                    ///  * Days (32 bits) since the UNIX epoch
                    ///
                    /// Generated from these locations:
                    /// * Table `Date` in the file `Schema.fbs:252`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct Date {
                        /// The field `unit` in the table `Date`
                        pub unit: self::DateUnit,
                    }

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for Date {
                        fn default() -> Self {
                            Self {
                                unit: self::DateUnit::Millisecond,
                            }
                        }
                    }

                    impl Date {
                        /// Creates a [DateBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> DateBuilder<()> {
                            DateBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            field_unit: impl ::planus::WriteAsDefault<self::DateUnit, self::DateUnit>,
                        ) -> ::planus::Offset<Self> {
                            let prepared_unit =
                                field_unit.prepare(builder, &self::DateUnit::Millisecond);

                            let mut table_writer: ::planus::table_writer::TableWriter<6> =
                                ::core::default::Default::default();
                            if prepared_unit.is_some() {
                                table_writer.write_entry::<self::DateUnit>(0);
                            }

                            unsafe {
                                table_writer.finish(builder, |object_writer| {
                                    if let ::core::option::Option::Some(prepared_unit) =
                                        prepared_unit
                                    {
                                        object_writer.write::<_, _, 2>(&prepared_unit);
                                    }
                                });
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Date>> for Date {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Date> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Date>> for Date {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Date>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Date> for Date {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Date> {
                            Date::create(builder, self.unit)
                        }
                    }

                    /// Builder for serializing an instance of the [Date] type.
                    ///
                    /// Can be created using the [Date::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct DateBuilder<State>(State);

                    impl DateBuilder<()> {
                        /// Setter for the [`unit` field](Date#structfield.unit).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn unit<T0>(self, value: T0) -> DateBuilder<(T0,)>
                        where
                            T0: ::planus::WriteAsDefault<self::DateUnit, self::DateUnit>,
                        {
                            DateBuilder((value,))
                        }

                        /// Sets the [`unit` field](Date#structfield.unit) to the default value.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn unit_as_default(self) -> DateBuilder<(::planus::DefaultValue,)> {
                            self.unit(::planus::DefaultValue)
                        }
                    }

                    impl<T0> DateBuilder<(T0,)> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Date].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Date>
                        where
                            Self: ::planus::WriteAsOffset<Date>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl<T0: ::planus::WriteAsDefault<self::DateUnit, self::DateUnit>>
                        ::planus::WriteAs<::planus::Offset<Date>> for DateBuilder<(T0,)>
                    {
                        type Prepared = ::planus::Offset<Date>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Date> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl<T0: ::planus::WriteAsDefault<self::DateUnit, self::DateUnit>>
                        ::planus::WriteAsOptional<::planus::Offset<Date>> for DateBuilder<(T0,)>
                    {
                        type Prepared = ::planus::Offset<Date>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Date>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl<T0: ::planus::WriteAsDefault<self::DateUnit, self::DateUnit>>
                        ::planus::WriteAsOffset<Date> for DateBuilder<(T0,)>
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Date> {
                            let (v0,) = &self.0;
                            Date::create(builder, v0)
                        }
                    }

                    /// Reference to a deserialized [Date].
                    #[derive(Copy, Clone)]
                    pub struct DateRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> DateRef<'a> {
                        /// Getter for the [`unit` field](Date#structfield.unit).
                        #[inline]
                        pub fn unit(&self) -> ::planus::Result<self::DateUnit> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(0, "Date", "unit")?
                                    .unwrap_or(self::DateUnit::Millisecond),
                            )
                        }
                    }

                    impl<'a> ::core::fmt::Debug for DateRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("DateRef");
                            f.field("unit", &self.unit());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<DateRef<'a>> for Date {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: DateRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                unit: ::core::convert::TryInto::try_into(value.unit()?)?,
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for DateRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for DateRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[DateRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<Date>> for Date {
                        type Value = ::planus::Offset<Date>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Date>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for DateRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[DateRef]", "read_as_root", 0)
                            })
                        }
                    }

                    /// The enum `TimeUnit` in the namespace `org.apache.arrow.flatbuf`
                    ///
                    /// Generated from these locations:
                    /// * Enum `TimeUnit` in the file `Schema.fbs:256`
                    #[derive(
                        Copy,
                        Clone,
                        Debug,
                        PartialEq,
                        Eq,
                        PartialOrd,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    #[repr(i16)]
                    pub enum TimeUnit {
                        /// The variant `SECOND` in the enum `TimeUnit`
                        Second = 0,

                        /// The variant `MILLISECOND` in the enum `TimeUnit`
                        Millisecond = 1,

                        /// The variant `MICROSECOND` in the enum `TimeUnit`
                        Microsecond = 2,

                        /// The variant `NANOSECOND` in the enum `TimeUnit`
                        Nanosecond = 3,
                    }

                    impl TimeUnit {
                        /// Array containing all valid variants of TimeUnit
                        pub const ENUM_VALUES: [Self; 4] = [
                            Self::Second,
                            Self::Millisecond,
                            Self::Microsecond,
                            Self::Nanosecond,
                        ];
                    }

                    impl ::core::convert::TryFrom<i16> for TimeUnit {
                        type Error = ::planus::errors::UnknownEnumTagKind;
                        #[inline]
                        fn try_from(
                            value: i16,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTagKind>
                        {
                            #[allow(clippy::match_single_binding)]
                            match value {
                                0 => ::core::result::Result::Ok(TimeUnit::Second),
                                1 => ::core::result::Result::Ok(TimeUnit::Millisecond),
                                2 => ::core::result::Result::Ok(TimeUnit::Microsecond),
                                3 => ::core::result::Result::Ok(TimeUnit::Nanosecond),

                                _ => ::core::result::Result::Err(
                                    ::planus::errors::UnknownEnumTagKind { tag: value as i128 },
                                ),
                            }
                        }
                    }

                    impl ::core::convert::From<TimeUnit> for i16 {
                        #[inline]
                        fn from(value: TimeUnit) -> Self {
                            value as i16
                        }
                    }

                    /// # Safety
                    /// The Planus compiler correctly calculates `ALIGNMENT` and `SIZE`.
                    unsafe impl ::planus::Primitive for TimeUnit {
                        const ALIGNMENT: usize = 2;
                        const SIZE: usize = 2;
                    }

                    impl ::planus::WriteAsPrimitive<TimeUnit> for TimeUnit {
                        #[inline]
                        fn write<const N: usize>(
                            &self,
                            cursor: ::planus::Cursor<'_, N>,
                            buffer_position: u32,
                        ) {
                            (*self as i16).write(cursor, buffer_position);
                        }
                    }

                    impl ::planus::WriteAs<TimeUnit> for TimeUnit {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> TimeUnit {
                            *self
                        }
                    }

                    impl ::planus::WriteAsDefault<TimeUnit, TimeUnit> for TimeUnit {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                            default: &TimeUnit,
                        ) -> ::core::option::Option<TimeUnit> {
                            if self == default {
                                ::core::option::Option::None
                            } else {
                                ::core::option::Option::Some(*self)
                            }
                        }
                    }

                    impl ::planus::WriteAsOptional<TimeUnit> for TimeUnit {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<TimeUnit> {
                            ::core::option::Option::Some(*self)
                        }
                    }

                    impl<'buf> ::planus::TableRead<'buf> for TimeUnit {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            let n: i16 = ::planus::TableRead::from_buffer(buffer, offset)?;
                            ::core::result::Result::Ok(::core::convert::TryInto::try_into(n)?)
                        }
                    }

                    impl<'buf> ::planus::VectorReadInner<'buf> for TimeUnit {
                        type Error = ::planus::errors::UnknownEnumTag;
                        const STRIDE: usize = 2;
                        #[inline]
                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTag>
                        {
                            let value = unsafe {
                                <i16 as ::planus::VectorRead>::from_buffer(buffer, offset)
                            };
                            let value: ::core::result::Result<Self, _> =
                                ::core::convert::TryInto::try_into(value);
                            value.map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "TimeUnit",
                                    "VectorRead::from_buffer",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<TimeUnit> for TimeUnit {
                        const STRIDE: usize = 2;

                        type Value = Self;

                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self {
                            *self
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[Self],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 2];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (2 * i) as u32,
                                );
                            }
                        }
                    }

                    ///  Time is either a 32-bit or 64-bit signed integer type representing an
                    ///  elapsed time since midnight, stored in either of four units: seconds,
                    ///  milliseconds, microseconds or nanoseconds.
                    ///
                    ///  The integer `bitWidth` depends on the `unit` and must be one of the following:
                    ///  * SECOND and MILLISECOND: 32 bits
                    ///  * MICROSECOND and NANOSECOND: 64 bits
                    ///
                    ///  The allowed values are between 0 (inclusive) and 86400 (=24*60*60) seconds
                    ///  (exclusive), adjusted for the time unit (for example, up to 86400000
                    ///  exclusive for the MILLISECOND unit).
                    ///  This definition doesn't allow for leap seconds. Time values from
                    ///  measurements with leap seconds will need to be corrected when ingesting
                    ///  into Arrow (for example by replacing the value 86400 with 86399).
                    ///
                    /// Generated from these locations:
                    /// * Table `Time` in the file `Schema.fbs:272`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct Time {
                        /// The field `unit` in the table `Time`
                        pub unit: self::TimeUnit,
                        /// The field `bitWidth` in the table `Time`
                        pub bit_width: i32,
                    }

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for Time {
                        fn default() -> Self {
                            Self {
                                unit: self::TimeUnit::Millisecond,
                                bit_width: 32,
                            }
                        }
                    }

                    impl Time {
                        /// Creates a [TimeBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> TimeBuilder<()> {
                            TimeBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            field_unit: impl ::planus::WriteAsDefault<self::TimeUnit, self::TimeUnit>,
                            field_bit_width: impl ::planus::WriteAsDefault<i32, i32>,
                        ) -> ::planus::Offset<Self> {
                            let prepared_unit =
                                field_unit.prepare(builder, &self::TimeUnit::Millisecond);
                            let prepared_bit_width = field_bit_width.prepare(builder, &32);

                            let mut table_writer: ::planus::table_writer::TableWriter<8> =
                                ::core::default::Default::default();
                            if prepared_bit_width.is_some() {
                                table_writer.write_entry::<i32>(1);
                            }
                            if prepared_unit.is_some() {
                                table_writer.write_entry::<self::TimeUnit>(0);
                            }

                            unsafe {
                                table_writer.finish(builder, |object_writer| {
                                    if let ::core::option::Option::Some(prepared_bit_width) =
                                        prepared_bit_width
                                    {
                                        object_writer.write::<_, _, 4>(&prepared_bit_width);
                                    }
                                    if let ::core::option::Option::Some(prepared_unit) =
                                        prepared_unit
                                    {
                                        object_writer.write::<_, _, 2>(&prepared_unit);
                                    }
                                });
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Time>> for Time {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Time> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Time>> for Time {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Time>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Time> for Time {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Time> {
                            Time::create(builder, self.unit, self.bit_width)
                        }
                    }

                    /// Builder for serializing an instance of the [Time] type.
                    ///
                    /// Can be created using the [Time::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct TimeBuilder<State>(State);

                    impl TimeBuilder<()> {
                        /// Setter for the [`unit` field](Time#structfield.unit).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn unit<T0>(self, value: T0) -> TimeBuilder<(T0,)>
                        where
                            T0: ::planus::WriteAsDefault<self::TimeUnit, self::TimeUnit>,
                        {
                            TimeBuilder((value,))
                        }

                        /// Sets the [`unit` field](Time#structfield.unit) to the default value.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn unit_as_default(self) -> TimeBuilder<(::planus::DefaultValue,)> {
                            self.unit(::planus::DefaultValue)
                        }
                    }

                    impl<T0> TimeBuilder<(T0,)> {
                        /// Setter for the [`bitWidth` field](Time#structfield.bit_width).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn bit_width<T1>(self, value: T1) -> TimeBuilder<(T0, T1)>
                        where
                            T1: ::planus::WriteAsDefault<i32, i32>,
                        {
                            let (v0,) = self.0;
                            TimeBuilder((v0, value))
                        }

                        /// Sets the [`bitWidth` field](Time#structfield.bit_width) to the default value.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn bit_width_as_default(
                            self,
                        ) -> TimeBuilder<(T0, ::planus::DefaultValue)> {
                            self.bit_width(::planus::DefaultValue)
                        }
                    }

                    impl<T0, T1> TimeBuilder<(T0, T1)> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Time].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Time>
                        where
                            Self: ::planus::WriteAsOffset<Time>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<self::TimeUnit, self::TimeUnit>,
                            T1: ::planus::WriteAsDefault<i32, i32>,
                        > ::planus::WriteAs<::planus::Offset<Time>> for TimeBuilder<(T0, T1)>
                    {
                        type Prepared = ::planus::Offset<Time>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Time> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<self::TimeUnit, self::TimeUnit>,
                            T1: ::planus::WriteAsDefault<i32, i32>,
                        > ::planus::WriteAsOptional<::planus::Offset<Time>>
                        for TimeBuilder<(T0, T1)>
                    {
                        type Prepared = ::planus::Offset<Time>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Time>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<self::TimeUnit, self::TimeUnit>,
                            T1: ::planus::WriteAsDefault<i32, i32>,
                        > ::planus::WriteAsOffset<Time> for TimeBuilder<(T0, T1)>
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Time> {
                            let (v0, v1) = &self.0;
                            Time::create(builder, v0, v1)
                        }
                    }

                    /// Reference to a deserialized [Time].
                    #[derive(Copy, Clone)]
                    pub struct TimeRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> TimeRef<'a> {
                        /// Getter for the [`unit` field](Time#structfield.unit).
                        #[inline]
                        pub fn unit(&self) -> ::planus::Result<self::TimeUnit> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(0, "Time", "unit")?
                                    .unwrap_or(self::TimeUnit::Millisecond),
                            )
                        }

                        /// Getter for the [`bitWidth` field](Time#structfield.bit_width).
                        #[inline]
                        pub fn bit_width(&self) -> ::planus::Result<i32> {
                            ::core::result::Result::Ok(
                                self.0.access(1, "Time", "bit_width")?.unwrap_or(32),
                            )
                        }
                    }

                    impl<'a> ::core::fmt::Debug for TimeRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("TimeRef");
                            f.field("unit", &self.unit());
                            f.field("bit_width", &self.bit_width());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<TimeRef<'a>> for Time {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: TimeRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                unit: ::core::convert::TryInto::try_into(value.unit()?)?,
                                bit_width: ::core::convert::TryInto::try_into(value.bit_width()?)?,
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for TimeRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for TimeRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[TimeRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<Time>> for Time {
                        type Value = ::planus::Offset<Time>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Time>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for TimeRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[TimeRef]", "read_as_root", 0)
                            })
                        }
                    }

                    ///  Timestamp is a 64-bit signed integer representing an elapsed time since a
                    ///  fixed epoch, stored in either of four units: seconds, milliseconds,
                    ///  microseconds or nanoseconds, and is optionally annotated with a timezone.
                    ///
                    ///  Timestamp values do not include any leap seconds (in other words, all
                    ///  days are considered 86400 seconds long).
                    ///
                    ///  Timestamps with a non-empty timezone
                    ///  ------------------------------------
                    ///
                    ///  If a Timestamp column has a non-empty timezone value, its epoch is
                    ///  1970-01-01 00:00:00 (January 1st 1970, midnight) in the *UTC* timezone
                    ///  (the Unix epoch), regardless of the Timestamp's own timezone.
                    ///
                    ///  Therefore, timestamp values with a non-empty timezone correspond to
                    ///  physical points in time together with some additional information about
                    ///  how the data was obtained and/or how to display it (the timezone).
                    ///
                    ///    For example, the timestamp value 0 with the timezone string "Europe/Paris"
                    ///    corresponds to "January 1st 1970, 00h00" in the UTC timezone, but the
                    ///    application may prefer to display it as "January 1st 1970, 01h00" in
                    ///    the Europe/Paris timezone (which is the same physical point in time).
                    ///
                    ///  One consequence is that timestamp values with a non-empty timezone
                    ///  can be compared and ordered directly, since they all share the same
                    ///  well-known point of reference (the Unix epoch).
                    ///
                    ///  Timestamps with an unset / empty timezone
                    ///  -----------------------------------------
                    ///
                    ///  If a Timestamp column has no timezone value, its epoch is
                    ///  1970-01-01 00:00:00 (January 1st 1970, midnight) in an *unknown* timezone.
                    ///
                    ///  Therefore, timestamp values without a timezone cannot be meaningfully
                    ///  interpreted as physical points in time, but only as calendar / clock
                    ///  indications ("wall clock time") in an unspecified timezone.
                    ///
                    ///    For example, the timestamp value 0 with an empty timezone string
                    ///    corresponds to "January 1st 1970, 00h00" in an unknown timezone: there
                    ///    is not enough information to interpret it as a well-defined physical
                    ///    point in time.
                    ///
                    ///  One consequence is that timestamp values without a timezone cannot
                    ///  be reliably compared or ordered, since they may have different points of
                    ///  reference.  In particular, it is *not* possible to interpret an unset
                    ///  or empty timezone as the same as "UTC".
                    ///
                    ///  Conversion between timezones
                    ///  ----------------------------
                    ///
                    ///  If a Timestamp column has a non-empty timezone, changing the timezone
                    ///  to a different non-empty value is a metadata-only operation:
                    ///  the timestamp values need not change as their point of reference remains
                    ///  the same (the Unix epoch).
                    ///
                    ///  However, if a Timestamp column has no timezone value, changing it to a
                    ///  non-empty value requires to think about the desired semantics.
                    ///  One possibility is to assume that the original timestamp values are
                    ///  relative to the epoch of the timezone being set; timestamp values should
                    ///  then adjusted to the Unix epoch (for example, changing the timezone from
                    ///  empty to "Europe/Paris" would require converting the timestamp values
                    ///  from "Europe/Paris" to "UTC", which seems counter-intuitive but is
                    ///  nevertheless correct).
                    ///
                    ///  Guidelines for encoding data from external libraries
                    ///  ----------------------------------------------------
                    ///
                    ///  Date & time libraries often have multiple different data types for temporal
                    ///  data. In order to ease interoperability between different implementations the
                    ///  Arrow project has some recommendations for encoding these types into a Timestamp
                    ///  column.
                    ///
                    ///  An "instant" represents a physical point in time that has no relevant timezone
                    ///  (for example, astronomical data). To encode an instant, use a Timestamp with
                    ///  the timezone string set to "UTC", and make sure the Timestamp values
                    ///  are relative to the UTC epoch (January 1st 1970, midnight).
                    ///
                    ///  A "zoned date-time" represents a physical point in time annotated with an
                    ///  informative timezone (for example, the timezone in which the data was
                    ///  recorded).  To encode a zoned date-time, use a Timestamp with the timezone
                    ///  string set to the name of the timezone, and make sure the Timestamp values
                    ///  are relative to the UTC epoch (January 1st 1970, midnight).
                    ///
                    ///   (There is some ambiguity between an instant and a zoned date-time with the
                    ///    UTC timezone.  Both of these are stored the same in Arrow.  Typically,
                    ///    this distinction does not matter.  If it does, then an application should
                    ///    use custom metadata or an extension type to distinguish between the two cases.)
                    ///
                    ///  An "offset date-time" represents a physical point in time combined with an
                    ///  explicit offset from UTC.  To encode an offset date-time, use a Timestamp
                    ///  with the timezone string set to the numeric timezone offset string
                    ///  (e.g. "+03:00"), and make sure the Timestamp values are relative to
                    ///  the UTC epoch (January 1st 1970, midnight).
                    ///
                    ///  A "naive date-time" (also called "local date-time" in some libraries)
                    ///  represents a wall clock time combined with a calendar date, but with
                    ///  no indication of how to map this information to a physical point in time.
                    ///  Naive date-times must be handled with care because of this missing
                    ///  information, and also because daylight saving time (DST) may make
                    ///  some values ambiguous or nonexistent. A naive date-time may be
                    ///  stored as a struct with Date and Time fields. However, it may also be
                    ///  encoded into a Timestamp column with an empty timezone. The timestamp
                    ///  values should be computed "as if" the timezone of the date-time values
                    ///  was UTC; for example, the naive date-time "January 1st 1970, 00h00" would
                    ///  be encoded as timestamp value 0.
                    ///
                    /// Generated from these locations:
                    /// * Table `Timestamp` in the file `Schema.fbs:382`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct Timestamp {
                        /// The field `unit` in the table `Timestamp`
                        pub unit: self::TimeUnit,
                        ///  The timezone is an optional string indicating the name of a timezone,
                        ///  one of:
                        ///
                        ///  * As used in the Olson timezone database (the "tz database" or
                        ///    "tzdata"), such as "America/New_York".
                        ///  * An absolute timezone offset of the form "+XX:XX" or "-XX:XX",
                        ///    such as "+07:30".
                        ///
                        ///  Whether a timezone string is present indicates different semantics about
                        ///  the data (see above).
                        pub timezone: ::core::option::Option<::planus::alloc::string::String>,
                    }

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for Timestamp {
                        fn default() -> Self {
                            Self {
                                unit: self::TimeUnit::Second,
                                timezone: ::core::default::Default::default(),
                            }
                        }
                    }

                    impl Timestamp {
                        /// Creates a [TimestampBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> TimestampBuilder<()> {
                            TimestampBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            field_unit: impl ::planus::WriteAsDefault<self::TimeUnit, self::TimeUnit>,
                            field_timezone: impl ::planus::WriteAsOptional<
                                ::planus::Offset<::core::primitive::str>,
                            >,
                        ) -> ::planus::Offset<Self> {
                            let prepared_unit =
                                field_unit.prepare(builder, &self::TimeUnit::Second);
                            let prepared_timezone = field_timezone.prepare(builder);

                            let mut table_writer: ::planus::table_writer::TableWriter<8> =
                                ::core::default::Default::default();
                            if prepared_timezone.is_some() {
                                table_writer.write_entry::<::planus::Offset<str>>(1);
                            }
                            if prepared_unit.is_some() {
                                table_writer.write_entry::<self::TimeUnit>(0);
                            }

                            unsafe {
                                table_writer.finish(builder, |object_writer| {
                                    if let ::core::option::Option::Some(prepared_timezone) =
                                        prepared_timezone
                                    {
                                        object_writer.write::<_, _, 4>(&prepared_timezone);
                                    }
                                    if let ::core::option::Option::Some(prepared_unit) =
                                        prepared_unit
                                    {
                                        object_writer.write::<_, _, 2>(&prepared_unit);
                                    }
                                });
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Timestamp>> for Timestamp {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Timestamp> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Timestamp>> for Timestamp {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Timestamp>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Timestamp> for Timestamp {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Timestamp> {
                            Timestamp::create(builder, self.unit, &self.timezone)
                        }
                    }

                    /// Builder for serializing an instance of the [Timestamp] type.
                    ///
                    /// Can be created using the [Timestamp::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct TimestampBuilder<State>(State);

                    impl TimestampBuilder<()> {
                        /// Setter for the [`unit` field](Timestamp#structfield.unit).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn unit<T0>(self, value: T0) -> TimestampBuilder<(T0,)>
                        where
                            T0: ::planus::WriteAsDefault<self::TimeUnit, self::TimeUnit>,
                        {
                            TimestampBuilder((value,))
                        }

                        /// Sets the [`unit` field](Timestamp#structfield.unit) to the default value.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn unit_as_default(
                            self,
                        ) -> TimestampBuilder<(::planus::DefaultValue,)> {
                            self.unit(::planus::DefaultValue)
                        }
                    }

                    impl<T0> TimestampBuilder<(T0,)> {
                        /// Setter for the [`timezone` field](Timestamp#structfield.timezone).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn timezone<T1>(self, value: T1) -> TimestampBuilder<(T0, T1)>
                        where
                            T1: ::planus::WriteAsOptional<::planus::Offset<::core::primitive::str>>,
                        {
                            let (v0,) = self.0;
                            TimestampBuilder((v0, value))
                        }

                        /// Sets the [`timezone` field](Timestamp#structfield.timezone) to null.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn timezone_as_null(self) -> TimestampBuilder<(T0, ())> {
                            self.timezone(())
                        }
                    }

                    impl<T0, T1> TimestampBuilder<(T0, T1)> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Timestamp].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Timestamp>
                        where
                            Self: ::planus::WriteAsOffset<Timestamp>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<self::TimeUnit, self::TimeUnit>,
                            T1: ::planus::WriteAsOptional<::planus::Offset<::core::primitive::str>>,
                        > ::planus::WriteAs<::planus::Offset<Timestamp>>
                        for TimestampBuilder<(T0, T1)>
                    {
                        type Prepared = ::planus::Offset<Timestamp>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Timestamp> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<self::TimeUnit, self::TimeUnit>,
                            T1: ::planus::WriteAsOptional<::planus::Offset<::core::primitive::str>>,
                        > ::planus::WriteAsOptional<::planus::Offset<Timestamp>>
                        for TimestampBuilder<(T0, T1)>
                    {
                        type Prepared = ::planus::Offset<Timestamp>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Timestamp>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<self::TimeUnit, self::TimeUnit>,
                            T1: ::planus::WriteAsOptional<::planus::Offset<::core::primitive::str>>,
                        > ::planus::WriteAsOffset<Timestamp> for TimestampBuilder<(T0, T1)>
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Timestamp> {
                            let (v0, v1) = &self.0;
                            Timestamp::create(builder, v0, v1)
                        }
                    }

                    /// Reference to a deserialized [Timestamp].
                    #[derive(Copy, Clone)]
                    pub struct TimestampRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> TimestampRef<'a> {
                        /// Getter for the [`unit` field](Timestamp#structfield.unit).
                        #[inline]
                        pub fn unit(&self) -> ::planus::Result<self::TimeUnit> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(0, "Timestamp", "unit")?
                                    .unwrap_or(self::TimeUnit::Second),
                            )
                        }

                        /// Getter for the [`timezone` field](Timestamp#structfield.timezone).
                        #[inline]
                        pub fn timezone(
                            &self,
                        ) -> ::planus::Result<::core::option::Option<&'a ::core::primitive::str>>
                        {
                            self.0.access(1, "Timestamp", "timezone")
                        }
                    }

                    impl<'a> ::core::fmt::Debug for TimestampRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("TimestampRef");
                            f.field("unit", &self.unit());
                            if let ::core::option::Option::Some(field_timezone) =
                                self.timezone().transpose()
                            {
                                f.field("timezone", &field_timezone);
                            }
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<TimestampRef<'a>> for Timestamp {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: TimestampRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                unit: ::core::convert::TryInto::try_into(value.unit()?)?,
                                timezone: value.timezone()?.map(::core::convert::Into::into),
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for TimestampRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for TimestampRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[TimestampRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<Timestamp>> for Timestamp {
                        type Value = ::planus::Offset<Timestamp>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Timestamp>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for TimestampRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[TimestampRef]", "read_as_root", 0)
                            })
                        }
                    }

                    /// The enum `IntervalUnit` in the namespace `org.apache.arrow.flatbuf`
                    ///
                    /// Generated from these locations:
                    /// * Enum `IntervalUnit` in the file `Schema.fbs:398`
                    #[derive(
                        Copy,
                        Clone,
                        Debug,
                        PartialEq,
                        Eq,
                        PartialOrd,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    #[repr(i16)]
                    pub enum IntervalUnit {
                        /// The variant `YEAR_MONTH` in the enum `IntervalUnit`
                        YearMonth = 0,

                        /// The variant `DAY_TIME` in the enum `IntervalUnit`
                        DayTime = 1,

                        /// The variant `MONTH_DAY_NANO` in the enum `IntervalUnit`
                        MonthDayNano = 2,
                    }

                    impl IntervalUnit {
                        /// Array containing all valid variants of IntervalUnit
                        pub const ENUM_VALUES: [Self; 3] =
                            [Self::YearMonth, Self::DayTime, Self::MonthDayNano];
                    }

                    impl ::core::convert::TryFrom<i16> for IntervalUnit {
                        type Error = ::planus::errors::UnknownEnumTagKind;
                        #[inline]
                        fn try_from(
                            value: i16,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTagKind>
                        {
                            #[allow(clippy::match_single_binding)]
                            match value {
                                0 => ::core::result::Result::Ok(IntervalUnit::YearMonth),
                                1 => ::core::result::Result::Ok(IntervalUnit::DayTime),
                                2 => ::core::result::Result::Ok(IntervalUnit::MonthDayNano),

                                _ => ::core::result::Result::Err(
                                    ::planus::errors::UnknownEnumTagKind { tag: value as i128 },
                                ),
                            }
                        }
                    }

                    impl ::core::convert::From<IntervalUnit> for i16 {
                        #[inline]
                        fn from(value: IntervalUnit) -> Self {
                            value as i16
                        }
                    }

                    /// # Safety
                    /// The Planus compiler correctly calculates `ALIGNMENT` and `SIZE`.
                    unsafe impl ::planus::Primitive for IntervalUnit {
                        const ALIGNMENT: usize = 2;
                        const SIZE: usize = 2;
                    }

                    impl ::planus::WriteAsPrimitive<IntervalUnit> for IntervalUnit {
                        #[inline]
                        fn write<const N: usize>(
                            &self,
                            cursor: ::planus::Cursor<'_, N>,
                            buffer_position: u32,
                        ) {
                            (*self as i16).write(cursor, buffer_position);
                        }
                    }

                    impl ::planus::WriteAs<IntervalUnit> for IntervalUnit {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> IntervalUnit {
                            *self
                        }
                    }

                    impl ::planus::WriteAsDefault<IntervalUnit, IntervalUnit> for IntervalUnit {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                            default: &IntervalUnit,
                        ) -> ::core::option::Option<IntervalUnit> {
                            if self == default {
                                ::core::option::Option::None
                            } else {
                                ::core::option::Option::Some(*self)
                            }
                        }
                    }

                    impl ::planus::WriteAsOptional<IntervalUnit> for IntervalUnit {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<IntervalUnit> {
                            ::core::option::Option::Some(*self)
                        }
                    }

                    impl<'buf> ::planus::TableRead<'buf> for IntervalUnit {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            let n: i16 = ::planus::TableRead::from_buffer(buffer, offset)?;
                            ::core::result::Result::Ok(::core::convert::TryInto::try_into(n)?)
                        }
                    }

                    impl<'buf> ::planus::VectorReadInner<'buf> for IntervalUnit {
                        type Error = ::planus::errors::UnknownEnumTag;
                        const STRIDE: usize = 2;
                        #[inline]
                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTag>
                        {
                            let value = unsafe {
                                <i16 as ::planus::VectorRead>::from_buffer(buffer, offset)
                            };
                            let value: ::core::result::Result<Self, _> =
                                ::core::convert::TryInto::try_into(value);
                            value.map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "IntervalUnit",
                                    "VectorRead::from_buffer",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<IntervalUnit> for IntervalUnit {
                        const STRIDE: usize = 2;

                        type Value = Self;

                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self {
                            *self
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[Self],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 2];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (2 * i) as u32,
                                );
                            }
                        }
                    }

                    /// The table `Interval` in the namespace `org.apache.arrow.flatbuf`
                    ///
                    /// Generated from these locations:
                    /// * Table `Interval` in the file `Schema.fbs:417`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct Interval {
                        /// The field `unit` in the table `Interval`
                        pub unit: self::IntervalUnit,
                    }

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for Interval {
                        fn default() -> Self {
                            Self {
                                unit: self::IntervalUnit::YearMonth,
                            }
                        }
                    }

                    impl Interval {
                        /// Creates a [IntervalBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> IntervalBuilder<()> {
                            IntervalBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            field_unit: impl ::planus::WriteAsDefault<
                                self::IntervalUnit,
                                self::IntervalUnit,
                            >,
                        ) -> ::planus::Offset<Self> {
                            let prepared_unit =
                                field_unit.prepare(builder, &self::IntervalUnit::YearMonth);

                            let mut table_writer: ::planus::table_writer::TableWriter<6> =
                                ::core::default::Default::default();
                            if prepared_unit.is_some() {
                                table_writer.write_entry::<self::IntervalUnit>(0);
                            }

                            unsafe {
                                table_writer.finish(builder, |object_writer| {
                                    if let ::core::option::Option::Some(prepared_unit) =
                                        prepared_unit
                                    {
                                        object_writer.write::<_, _, 2>(&prepared_unit);
                                    }
                                });
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Interval>> for Interval {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Interval> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Interval>> for Interval {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Interval>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Interval> for Interval {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Interval> {
                            Interval::create(builder, self.unit)
                        }
                    }

                    /// Builder for serializing an instance of the [Interval] type.
                    ///
                    /// Can be created using the [Interval::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct IntervalBuilder<State>(State);

                    impl IntervalBuilder<()> {
                        /// Setter for the [`unit` field](Interval#structfield.unit).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn unit<T0>(self, value: T0) -> IntervalBuilder<(T0,)>
                        where
                            T0: ::planus::WriteAsDefault<self::IntervalUnit, self::IntervalUnit>,
                        {
                            IntervalBuilder((value,))
                        }

                        /// Sets the [`unit` field](Interval#structfield.unit) to the default value.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn unit_as_default(self) -> IntervalBuilder<(::planus::DefaultValue,)> {
                            self.unit(::planus::DefaultValue)
                        }
                    }

                    impl<T0> IntervalBuilder<(T0,)> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Interval].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Interval>
                        where
                            Self: ::planus::WriteAsOffset<Interval>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl<T0: ::planus::WriteAsDefault<self::IntervalUnit, self::IntervalUnit>>
                        ::planus::WriteAs<::planus::Offset<Interval>> for IntervalBuilder<(T0,)>
                    {
                        type Prepared = ::planus::Offset<Interval>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Interval> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl<T0: ::planus::WriteAsDefault<self::IntervalUnit, self::IntervalUnit>>
                        ::planus::WriteAsOptional<::planus::Offset<Interval>>
                        for IntervalBuilder<(T0,)>
                    {
                        type Prepared = ::planus::Offset<Interval>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Interval>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl<T0: ::planus::WriteAsDefault<self::IntervalUnit, self::IntervalUnit>>
                        ::planus::WriteAsOffset<Interval> for IntervalBuilder<(T0,)>
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Interval> {
                            let (v0,) = &self.0;
                            Interval::create(builder, v0)
                        }
                    }

                    /// Reference to a deserialized [Interval].
                    #[derive(Copy, Clone)]
                    pub struct IntervalRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> IntervalRef<'a> {
                        /// Getter for the [`unit` field](Interval#structfield.unit).
                        #[inline]
                        pub fn unit(&self) -> ::planus::Result<self::IntervalUnit> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(0, "Interval", "unit")?
                                    .unwrap_or(self::IntervalUnit::YearMonth),
                            )
                        }
                    }

                    impl<'a> ::core::fmt::Debug for IntervalRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("IntervalRef");
                            f.field("unit", &self.unit());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<IntervalRef<'a>> for Interval {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: IntervalRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                unit: ::core::convert::TryInto::try_into(value.unit()?)?,
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for IntervalRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for IntervalRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[IntervalRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<Interval>> for Interval {
                        type Value = ::planus::Offset<Interval>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Interval>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for IntervalRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[IntervalRef]", "read_as_root", 0)
                            })
                        }
                    }

                    /// The table `Duration` in the namespace `org.apache.arrow.flatbuf`
                    ///
                    /// Generated from these locations:
                    /// * Table `Duration` in the file `Schema.fbs:434`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct Duration {
                        /// The field `unit` in the table `Duration`
                        pub unit: self::TimeUnit,
                    }

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for Duration {
                        fn default() -> Self {
                            Self {
                                unit: self::TimeUnit::Millisecond,
                            }
                        }
                    }

                    impl Duration {
                        /// Creates a [DurationBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> DurationBuilder<()> {
                            DurationBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            field_unit: impl ::planus::WriteAsDefault<self::TimeUnit, self::TimeUnit>,
                        ) -> ::planus::Offset<Self> {
                            let prepared_unit =
                                field_unit.prepare(builder, &self::TimeUnit::Millisecond);

                            let mut table_writer: ::planus::table_writer::TableWriter<6> =
                                ::core::default::Default::default();
                            if prepared_unit.is_some() {
                                table_writer.write_entry::<self::TimeUnit>(0);
                            }

                            unsafe {
                                table_writer.finish(builder, |object_writer| {
                                    if let ::core::option::Option::Some(prepared_unit) =
                                        prepared_unit
                                    {
                                        object_writer.write::<_, _, 2>(&prepared_unit);
                                    }
                                });
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Duration>> for Duration {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Duration> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Duration>> for Duration {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Duration>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Duration> for Duration {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Duration> {
                            Duration::create(builder, self.unit)
                        }
                    }

                    /// Builder for serializing an instance of the [Duration] type.
                    ///
                    /// Can be created using the [Duration::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct DurationBuilder<State>(State);

                    impl DurationBuilder<()> {
                        /// Setter for the [`unit` field](Duration#structfield.unit).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn unit<T0>(self, value: T0) -> DurationBuilder<(T0,)>
                        where
                            T0: ::planus::WriteAsDefault<self::TimeUnit, self::TimeUnit>,
                        {
                            DurationBuilder((value,))
                        }

                        /// Sets the [`unit` field](Duration#structfield.unit) to the default value.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn unit_as_default(self) -> DurationBuilder<(::planus::DefaultValue,)> {
                            self.unit(::planus::DefaultValue)
                        }
                    }

                    impl<T0> DurationBuilder<(T0,)> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Duration].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Duration>
                        where
                            Self: ::planus::WriteAsOffset<Duration>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl<T0: ::planus::WriteAsDefault<self::TimeUnit, self::TimeUnit>>
                        ::planus::WriteAs<::planus::Offset<Duration>> for DurationBuilder<(T0,)>
                    {
                        type Prepared = ::planus::Offset<Duration>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Duration> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl<T0: ::planus::WriteAsDefault<self::TimeUnit, self::TimeUnit>>
                        ::planus::WriteAsOptional<::planus::Offset<Duration>>
                        for DurationBuilder<(T0,)>
                    {
                        type Prepared = ::planus::Offset<Duration>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Duration>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl<T0: ::planus::WriteAsDefault<self::TimeUnit, self::TimeUnit>>
                        ::planus::WriteAsOffset<Duration> for DurationBuilder<(T0,)>
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Duration> {
                            let (v0,) = &self.0;
                            Duration::create(builder, v0)
                        }
                    }

                    /// Reference to a deserialized [Duration].
                    #[derive(Copy, Clone)]
                    pub struct DurationRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> DurationRef<'a> {
                        /// Getter for the [`unit` field](Duration#structfield.unit).
                        #[inline]
                        pub fn unit(&self) -> ::planus::Result<self::TimeUnit> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(0, "Duration", "unit")?
                                    .unwrap_or(self::TimeUnit::Millisecond),
                            )
                        }
                    }

                    impl<'a> ::core::fmt::Debug for DurationRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("DurationRef");
                            f.field("unit", &self.unit());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<DurationRef<'a>> for Duration {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: DurationRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                unit: ::core::convert::TryInto::try_into(value.unit()?)?,
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for DurationRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for DurationRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[DurationRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<Duration>> for Duration {
                        type Value = ::planus::Offset<Duration>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Duration>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for DurationRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[DurationRef]", "read_as_root", 0)
                            })
                        }
                    }

                    ///  ----------------------------------------------------------------------
                    ///  Top-level Type value, enabling extensible type-specific metadata. We can
                    ///  add new logical types to Type without breaking backwards compatibility
                    ///
                    /// Generated from these locations:
                    /// * Union `Type` in the file `Schema.fbs:442`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub enum Type {
                        /// The variant of type `Null` in the union `Type`
                        Null(::planus::alloc::boxed::Box<self::Null>),

                        /// The variant of type `Int` in the union `Type`
                        Int(::planus::alloc::boxed::Box<self::Int>),

                        /// The variant of type `FloatingPoint` in the union `Type`
                        FloatingPoint(::planus::alloc::boxed::Box<self::FloatingPoint>),

                        /// The variant of type `Binary` in the union `Type`
                        Binary(::planus::alloc::boxed::Box<self::Binary>),

                        /// The variant of type `Utf8` in the union `Type`
                        Utf8(::planus::alloc::boxed::Box<self::Utf8>),

                        /// The variant of type `Bool` in the union `Type`
                        Bool(::planus::alloc::boxed::Box<self::Bool>),

                        /// The variant of type `Decimal` in the union `Type`
                        Decimal(::planus::alloc::boxed::Box<self::Decimal>),

                        /// The variant of type `Date` in the union `Type`
                        Date(::planus::alloc::boxed::Box<self::Date>),

                        /// The variant of type `Time` in the union `Type`
                        Time(::planus::alloc::boxed::Box<self::Time>),

                        /// The variant of type `Timestamp` in the union `Type`
                        Timestamp(::planus::alloc::boxed::Box<self::Timestamp>),

                        /// The variant of type `Interval` in the union `Type`
                        Interval(::planus::alloc::boxed::Box<self::Interval>),

                        /// The variant of type `List` in the union `Type`
                        List(::planus::alloc::boxed::Box<self::List>),

                        /// The variant of type `Struct_` in the union `Type`
                        Struct(::planus::alloc::boxed::Box<self::Struct>),

                        /// The variant of type `Union` in the union `Type`
                        Union(::planus::alloc::boxed::Box<self::Union>),

                        /// The variant of type `FixedSizeBinary` in the union `Type`
                        FixedSizeBinary(::planus::alloc::boxed::Box<self::FixedSizeBinary>),

                        /// The variant of type `FixedSizeList` in the union `Type`
                        FixedSizeList(::planus::alloc::boxed::Box<self::FixedSizeList>),

                        /// The variant of type `Map` in the union `Type`
                        Map(::planus::alloc::boxed::Box<self::Map>),

                        /// The variant of type `Duration` in the union `Type`
                        Duration(::planus::alloc::boxed::Box<self::Duration>),

                        /// The variant of type `LargeBinary` in the union `Type`
                        LargeBinary(::planus::alloc::boxed::Box<self::LargeBinary>),

                        /// The variant of type `LargeUtf8` in the union `Type`
                        LargeUtf8(::planus::alloc::boxed::Box<self::LargeUtf8>),

                        /// The variant of type `LargeList` in the union `Type`
                        LargeList(::planus::alloc::boxed::Box<self::LargeList>),

                        /// The variant of type `RunEndEncoded` in the union `Type`
                        RunEndEncoded(::planus::alloc::boxed::Box<self::RunEndEncoded>),

                        /// The variant of type `BinaryView` in the union `Type`
                        BinaryView(::planus::alloc::boxed::Box<self::BinaryView>),

                        /// The variant of type `Utf8View` in the union `Type`
                        Utf8View(::planus::alloc::boxed::Box<self::Utf8View>),

                        /// The variant of type `ListView` in the union `Type`
                        ListView(::planus::alloc::boxed::Box<self::ListView>),

                        /// The variant of type `LargeListView` in the union `Type`
                        LargeListView(::planus::alloc::boxed::Box<self::LargeListView>),
                    }

                    impl Type {
                        /// Creates a [TypeBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> TypeBuilder<::planus::Uninitialized> {
                            TypeBuilder(::planus::Uninitialized)
                        }

                        #[inline]
                        pub fn create_null(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::Null>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(1, value.prepare(builder).downcast())
                        }

                        #[inline]
                        pub fn create_int(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::Int>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(2, value.prepare(builder).downcast())
                        }

                        #[inline]
                        pub fn create_floating_point(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::FloatingPoint>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(3, value.prepare(builder).downcast())
                        }

                        #[inline]
                        pub fn create_binary(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::Binary>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(4, value.prepare(builder).downcast())
                        }

                        #[inline]
                        pub fn create_utf8(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::Utf8>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(5, value.prepare(builder).downcast())
                        }

                        #[inline]
                        pub fn create_bool(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::Bool>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(6, value.prepare(builder).downcast())
                        }

                        #[inline]
                        pub fn create_decimal(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::Decimal>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(7, value.prepare(builder).downcast())
                        }

                        #[inline]
                        pub fn create_date(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::Date>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(8, value.prepare(builder).downcast())
                        }

                        #[inline]
                        pub fn create_time(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::Time>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(9, value.prepare(builder).downcast())
                        }

                        #[inline]
                        pub fn create_timestamp(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::Timestamp>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(10, value.prepare(builder).downcast())
                        }

                        #[inline]
                        pub fn create_interval(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::Interval>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(11, value.prepare(builder).downcast())
                        }

                        #[inline]
                        pub fn create_list(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::List>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(12, value.prepare(builder).downcast())
                        }

                        #[inline]
                        pub fn create_struct(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::Struct>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(13, value.prepare(builder).downcast())
                        }

                        #[inline]
                        pub fn create_union(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::Union>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(14, value.prepare(builder).downcast())
                        }

                        #[inline]
                        pub fn create_fixed_size_binary(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::FixedSizeBinary>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(15, value.prepare(builder).downcast())
                        }

                        #[inline]
                        pub fn create_fixed_size_list(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::FixedSizeList>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(16, value.prepare(builder).downcast())
                        }

                        #[inline]
                        pub fn create_map(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::Map>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(17, value.prepare(builder).downcast())
                        }

                        #[inline]
                        pub fn create_duration(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::Duration>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(18, value.prepare(builder).downcast())
                        }

                        #[inline]
                        pub fn create_large_binary(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::LargeBinary>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(19, value.prepare(builder).downcast())
                        }

                        #[inline]
                        pub fn create_large_utf8(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::LargeUtf8>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(20, value.prepare(builder).downcast())
                        }

                        #[inline]
                        pub fn create_large_list(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::LargeList>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(21, value.prepare(builder).downcast())
                        }

                        #[inline]
                        pub fn create_run_end_encoded(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::RunEndEncoded>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(22, value.prepare(builder).downcast())
                        }

                        #[inline]
                        pub fn create_binary_view(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::BinaryView>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(23, value.prepare(builder).downcast())
                        }

                        #[inline]
                        pub fn create_utf8_view(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::Utf8View>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(24, value.prepare(builder).downcast())
                        }

                        #[inline]
                        pub fn create_list_view(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::ListView>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(25, value.prepare(builder).downcast())
                        }

                        #[inline]
                        pub fn create_large_list_view(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::LargeListView>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(26, value.prepare(builder).downcast())
                        }
                    }

                    impl ::planus::WriteAsUnion<Type> for Type {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<Self> {
                            match self {
                                Self::Null(value) => Self::create_null(builder, value),
                                Self::Int(value) => Self::create_int(builder, value),
                                Self::FloatingPoint(value) => {
                                    Self::create_floating_point(builder, value)
                                }
                                Self::Binary(value) => Self::create_binary(builder, value),
                                Self::Utf8(value) => Self::create_utf8(builder, value),
                                Self::Bool(value) => Self::create_bool(builder, value),
                                Self::Decimal(value) => Self::create_decimal(builder, value),
                                Self::Date(value) => Self::create_date(builder, value),
                                Self::Time(value) => Self::create_time(builder, value),
                                Self::Timestamp(value) => Self::create_timestamp(builder, value),
                                Self::Interval(value) => Self::create_interval(builder, value),
                                Self::List(value) => Self::create_list(builder, value),
                                Self::Struct(value) => Self::create_struct(builder, value),
                                Self::Union(value) => Self::create_union(builder, value),
                                Self::FixedSizeBinary(value) => {
                                    Self::create_fixed_size_binary(builder, value)
                                }
                                Self::FixedSizeList(value) => {
                                    Self::create_fixed_size_list(builder, value)
                                }
                                Self::Map(value) => Self::create_map(builder, value),
                                Self::Duration(value) => Self::create_duration(builder, value),
                                Self::LargeBinary(value) => {
                                    Self::create_large_binary(builder, value)
                                }
                                Self::LargeUtf8(value) => Self::create_large_utf8(builder, value),
                                Self::LargeList(value) => Self::create_large_list(builder, value),
                                Self::RunEndEncoded(value) => {
                                    Self::create_run_end_encoded(builder, value)
                                }
                                Self::BinaryView(value) => Self::create_binary_view(builder, value),
                                Self::Utf8View(value) => Self::create_utf8_view(builder, value),
                                Self::ListView(value) => Self::create_list_view(builder, value),
                                Self::LargeListView(value) => {
                                    Self::create_large_list_view(builder, value)
                                }
                            }
                        }
                    }

                    impl ::planus::WriteAsOptionalUnion<Type> for Type {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<Self>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }

                    /// Builder for serializing an instance of the [Type] type.
                    ///
                    /// Can be created using the [Type::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct TypeBuilder<T>(T);

                    impl TypeBuilder<::planus::Uninitialized> {
                        /// Creates an instance of the [`Null` variant](Type#variant.Null).
                        #[inline]
                        pub fn null<T>(self, value: T) -> TypeBuilder<::planus::Initialized<1, T>>
                        where
                            T: ::planus::WriteAsOffset<self::Null>,
                        {
                            TypeBuilder(::planus::Initialized(value))
                        }

                        /// Creates an instance of the [`Int` variant](Type#variant.Int).
                        #[inline]
                        pub fn int<T>(self, value: T) -> TypeBuilder<::planus::Initialized<2, T>>
                        where
                            T: ::planus::WriteAsOffset<self::Int>,
                        {
                            TypeBuilder(::planus::Initialized(value))
                        }

                        /// Creates an instance of the [`FloatingPoint` variant](Type#variant.FloatingPoint).
                        #[inline]
                        pub fn floating_point<T>(
                            self,
                            value: T,
                        ) -> TypeBuilder<::planus::Initialized<3, T>>
                        where
                            T: ::planus::WriteAsOffset<self::FloatingPoint>,
                        {
                            TypeBuilder(::planus::Initialized(value))
                        }

                        /// Creates an instance of the [`Binary` variant](Type#variant.Binary).
                        #[inline]
                        pub fn binary<T>(self, value: T) -> TypeBuilder<::planus::Initialized<4, T>>
                        where
                            T: ::planus::WriteAsOffset<self::Binary>,
                        {
                            TypeBuilder(::planus::Initialized(value))
                        }

                        /// Creates an instance of the [`Utf8` variant](Type#variant.Utf8).
                        #[inline]
                        pub fn utf8<T>(self, value: T) -> TypeBuilder<::planus::Initialized<5, T>>
                        where
                            T: ::planus::WriteAsOffset<self::Utf8>,
                        {
                            TypeBuilder(::planus::Initialized(value))
                        }

                        /// Creates an instance of the [`Bool` variant](Type#variant.Bool).
                        #[inline]
                        pub fn bool<T>(self, value: T) -> TypeBuilder<::planus::Initialized<6, T>>
                        where
                            T: ::planus::WriteAsOffset<self::Bool>,
                        {
                            TypeBuilder(::planus::Initialized(value))
                        }

                        /// Creates an instance of the [`Decimal` variant](Type#variant.Decimal).
                        #[inline]
                        pub fn decimal<T>(
                            self,
                            value: T,
                        ) -> TypeBuilder<::planus::Initialized<7, T>>
                        where
                            T: ::planus::WriteAsOffset<self::Decimal>,
                        {
                            TypeBuilder(::planus::Initialized(value))
                        }

                        /// Creates an instance of the [`Date` variant](Type#variant.Date).
                        #[inline]
                        pub fn date<T>(self, value: T) -> TypeBuilder<::planus::Initialized<8, T>>
                        where
                            T: ::planus::WriteAsOffset<self::Date>,
                        {
                            TypeBuilder(::planus::Initialized(value))
                        }

                        /// Creates an instance of the [`Time` variant](Type#variant.Time).
                        #[inline]
                        pub fn time<T>(self, value: T) -> TypeBuilder<::planus::Initialized<9, T>>
                        where
                            T: ::planus::WriteAsOffset<self::Time>,
                        {
                            TypeBuilder(::planus::Initialized(value))
                        }

                        /// Creates an instance of the [`Timestamp` variant](Type#variant.Timestamp).
                        #[inline]
                        pub fn timestamp<T>(
                            self,
                            value: T,
                        ) -> TypeBuilder<::planus::Initialized<10, T>>
                        where
                            T: ::planus::WriteAsOffset<self::Timestamp>,
                        {
                            TypeBuilder(::planus::Initialized(value))
                        }

                        /// Creates an instance of the [`Interval` variant](Type#variant.Interval).
                        #[inline]
                        pub fn interval<T>(
                            self,
                            value: T,
                        ) -> TypeBuilder<::planus::Initialized<11, T>>
                        where
                            T: ::planus::WriteAsOffset<self::Interval>,
                        {
                            TypeBuilder(::planus::Initialized(value))
                        }

                        /// Creates an instance of the [`List` variant](Type#variant.List).
                        #[inline]
                        pub fn list<T>(self, value: T) -> TypeBuilder<::planus::Initialized<12, T>>
                        where
                            T: ::planus::WriteAsOffset<self::List>,
                        {
                            TypeBuilder(::planus::Initialized(value))
                        }

                        /// Creates an instance of the [`Struct_` variant](Type#variant.Struct).
                        #[inline]
                        pub fn struct_<T>(
                            self,
                            value: T,
                        ) -> TypeBuilder<::planus::Initialized<13, T>>
                        where
                            T: ::planus::WriteAsOffset<self::Struct>,
                        {
                            TypeBuilder(::planus::Initialized(value))
                        }

                        /// Creates an instance of the [`Union` variant](Type#variant.Union).
                        #[inline]
                        pub fn union_<T>(
                            self,
                            value: T,
                        ) -> TypeBuilder<::planus::Initialized<14, T>>
                        where
                            T: ::planus::WriteAsOffset<self::Union>,
                        {
                            TypeBuilder(::planus::Initialized(value))
                        }

                        /// Creates an instance of the [`FixedSizeBinary` variant](Type#variant.FixedSizeBinary).
                        #[inline]
                        pub fn fixed_size_binary<T>(
                            self,
                            value: T,
                        ) -> TypeBuilder<::planus::Initialized<15, T>>
                        where
                            T: ::planus::WriteAsOffset<self::FixedSizeBinary>,
                        {
                            TypeBuilder(::planus::Initialized(value))
                        }

                        /// Creates an instance of the [`FixedSizeList` variant](Type#variant.FixedSizeList).
                        #[inline]
                        pub fn fixed_size_list<T>(
                            self,
                            value: T,
                        ) -> TypeBuilder<::planus::Initialized<16, T>>
                        where
                            T: ::planus::WriteAsOffset<self::FixedSizeList>,
                        {
                            TypeBuilder(::planus::Initialized(value))
                        }

                        /// Creates an instance of the [`Map` variant](Type#variant.Map).
                        #[inline]
                        pub fn map<T>(self, value: T) -> TypeBuilder<::planus::Initialized<17, T>>
                        where
                            T: ::planus::WriteAsOffset<self::Map>,
                        {
                            TypeBuilder(::planus::Initialized(value))
                        }

                        /// Creates an instance of the [`Duration` variant](Type#variant.Duration).
                        #[inline]
                        pub fn duration<T>(
                            self,
                            value: T,
                        ) -> TypeBuilder<::planus::Initialized<18, T>>
                        where
                            T: ::planus::WriteAsOffset<self::Duration>,
                        {
                            TypeBuilder(::planus::Initialized(value))
                        }

                        /// Creates an instance of the [`LargeBinary` variant](Type#variant.LargeBinary).
                        #[inline]
                        pub fn large_binary<T>(
                            self,
                            value: T,
                        ) -> TypeBuilder<::planus::Initialized<19, T>>
                        where
                            T: ::planus::WriteAsOffset<self::LargeBinary>,
                        {
                            TypeBuilder(::planus::Initialized(value))
                        }

                        /// Creates an instance of the [`LargeUtf8` variant](Type#variant.LargeUtf8).
                        #[inline]
                        pub fn large_utf8<T>(
                            self,
                            value: T,
                        ) -> TypeBuilder<::planus::Initialized<20, T>>
                        where
                            T: ::planus::WriteAsOffset<self::LargeUtf8>,
                        {
                            TypeBuilder(::planus::Initialized(value))
                        }

                        /// Creates an instance of the [`LargeList` variant](Type#variant.LargeList).
                        #[inline]
                        pub fn large_list<T>(
                            self,
                            value: T,
                        ) -> TypeBuilder<::planus::Initialized<21, T>>
                        where
                            T: ::planus::WriteAsOffset<self::LargeList>,
                        {
                            TypeBuilder(::planus::Initialized(value))
                        }

                        /// Creates an instance of the [`RunEndEncoded` variant](Type#variant.RunEndEncoded).
                        #[inline]
                        pub fn run_end_encoded<T>(
                            self,
                            value: T,
                        ) -> TypeBuilder<::planus::Initialized<22, T>>
                        where
                            T: ::planus::WriteAsOffset<self::RunEndEncoded>,
                        {
                            TypeBuilder(::planus::Initialized(value))
                        }

                        /// Creates an instance of the [`BinaryView` variant](Type#variant.BinaryView).
                        #[inline]
                        pub fn binary_view<T>(
                            self,
                            value: T,
                        ) -> TypeBuilder<::planus::Initialized<23, T>>
                        where
                            T: ::planus::WriteAsOffset<self::BinaryView>,
                        {
                            TypeBuilder(::planus::Initialized(value))
                        }

                        /// Creates an instance of the [`Utf8View` variant](Type#variant.Utf8View).
                        #[inline]
                        pub fn utf8_view<T>(
                            self,
                            value: T,
                        ) -> TypeBuilder<::planus::Initialized<24, T>>
                        where
                            T: ::planus::WriteAsOffset<self::Utf8View>,
                        {
                            TypeBuilder(::planus::Initialized(value))
                        }

                        /// Creates an instance of the [`ListView` variant](Type#variant.ListView).
                        #[inline]
                        pub fn list_view<T>(
                            self,
                            value: T,
                        ) -> TypeBuilder<::planus::Initialized<25, T>>
                        where
                            T: ::planus::WriteAsOffset<self::ListView>,
                        {
                            TypeBuilder(::planus::Initialized(value))
                        }

                        /// Creates an instance of the [`LargeListView` variant](Type#variant.LargeListView).
                        #[inline]
                        pub fn large_list_view<T>(
                            self,
                            value: T,
                        ) -> TypeBuilder<::planus::Initialized<26, T>>
                        where
                            T: ::planus::WriteAsOffset<self::LargeListView>,
                        {
                            TypeBuilder(::planus::Initialized(value))
                        }
                    }

                    impl<const N: u8, T> TypeBuilder<::planus::Initialized<N, T>> {
                        /// Finish writing the builder to get an [UnionOffset](::planus::UnionOffset) to a serialized [Type].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<Type>
                        where
                            Self: ::planus::WriteAsUnion<Type>,
                        {
                            ::planus::WriteAsUnion::prepare(&self, builder)
                        }
                    }

                    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<1, T>>
                    where
                        T: ::planus::WriteAsOffset<self::Null>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<Type> {
                            ::planus::UnionOffset::new(1, (self.0).0.prepare(builder).downcast())
                        }
                    }

                    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<1, T>>
                    where
                        T: ::planus::WriteAsOffset<self::Null>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<Type>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }
                    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<2, T>>
                    where
                        T: ::planus::WriteAsOffset<self::Int>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<Type> {
                            ::planus::UnionOffset::new(2, (self.0).0.prepare(builder).downcast())
                        }
                    }

                    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<2, T>>
                    where
                        T: ::planus::WriteAsOffset<self::Int>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<Type>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }
                    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<3, T>>
                    where
                        T: ::planus::WriteAsOffset<self::FloatingPoint>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<Type> {
                            ::planus::UnionOffset::new(3, (self.0).0.prepare(builder).downcast())
                        }
                    }

                    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<3, T>>
                    where
                        T: ::planus::WriteAsOffset<self::FloatingPoint>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<Type>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }
                    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<4, T>>
                    where
                        T: ::planus::WriteAsOffset<self::Binary>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<Type> {
                            ::planus::UnionOffset::new(4, (self.0).0.prepare(builder).downcast())
                        }
                    }

                    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<4, T>>
                    where
                        T: ::planus::WriteAsOffset<self::Binary>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<Type>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }
                    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<5, T>>
                    where
                        T: ::planus::WriteAsOffset<self::Utf8>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<Type> {
                            ::planus::UnionOffset::new(5, (self.0).0.prepare(builder).downcast())
                        }
                    }

                    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<5, T>>
                    where
                        T: ::planus::WriteAsOffset<self::Utf8>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<Type>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }
                    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<6, T>>
                    where
                        T: ::planus::WriteAsOffset<self::Bool>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<Type> {
                            ::planus::UnionOffset::new(6, (self.0).0.prepare(builder).downcast())
                        }
                    }

                    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<6, T>>
                    where
                        T: ::planus::WriteAsOffset<self::Bool>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<Type>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }
                    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<7, T>>
                    where
                        T: ::planus::WriteAsOffset<self::Decimal>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<Type> {
                            ::planus::UnionOffset::new(7, (self.0).0.prepare(builder).downcast())
                        }
                    }

                    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<7, T>>
                    where
                        T: ::planus::WriteAsOffset<self::Decimal>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<Type>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }
                    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<8, T>>
                    where
                        T: ::planus::WriteAsOffset<self::Date>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<Type> {
                            ::planus::UnionOffset::new(8, (self.0).0.prepare(builder).downcast())
                        }
                    }

                    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<8, T>>
                    where
                        T: ::planus::WriteAsOffset<self::Date>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<Type>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }
                    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<9, T>>
                    where
                        T: ::planus::WriteAsOffset<self::Time>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<Type> {
                            ::planus::UnionOffset::new(9, (self.0).0.prepare(builder).downcast())
                        }
                    }

                    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<9, T>>
                    where
                        T: ::planus::WriteAsOffset<self::Time>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<Type>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }
                    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<10, T>>
                    where
                        T: ::planus::WriteAsOffset<self::Timestamp>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<Type> {
                            ::planus::UnionOffset::new(10, (self.0).0.prepare(builder).downcast())
                        }
                    }

                    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<10, T>>
                    where
                        T: ::planus::WriteAsOffset<self::Timestamp>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<Type>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }
                    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<11, T>>
                    where
                        T: ::planus::WriteAsOffset<self::Interval>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<Type> {
                            ::planus::UnionOffset::new(11, (self.0).0.prepare(builder).downcast())
                        }
                    }

                    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<11, T>>
                    where
                        T: ::planus::WriteAsOffset<self::Interval>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<Type>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }
                    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<12, T>>
                    where
                        T: ::planus::WriteAsOffset<self::List>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<Type> {
                            ::planus::UnionOffset::new(12, (self.0).0.prepare(builder).downcast())
                        }
                    }

                    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<12, T>>
                    where
                        T: ::planus::WriteAsOffset<self::List>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<Type>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }
                    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<13, T>>
                    where
                        T: ::planus::WriteAsOffset<self::Struct>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<Type> {
                            ::planus::UnionOffset::new(13, (self.0).0.prepare(builder).downcast())
                        }
                    }

                    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<13, T>>
                    where
                        T: ::planus::WriteAsOffset<self::Struct>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<Type>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }
                    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<14, T>>
                    where
                        T: ::planus::WriteAsOffset<self::Union>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<Type> {
                            ::planus::UnionOffset::new(14, (self.0).0.prepare(builder).downcast())
                        }
                    }

                    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<14, T>>
                    where
                        T: ::planus::WriteAsOffset<self::Union>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<Type>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }
                    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<15, T>>
                    where
                        T: ::planus::WriteAsOffset<self::FixedSizeBinary>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<Type> {
                            ::planus::UnionOffset::new(15, (self.0).0.prepare(builder).downcast())
                        }
                    }

                    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<15, T>>
                    where
                        T: ::planus::WriteAsOffset<self::FixedSizeBinary>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<Type>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }
                    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<16, T>>
                    where
                        T: ::planus::WriteAsOffset<self::FixedSizeList>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<Type> {
                            ::planus::UnionOffset::new(16, (self.0).0.prepare(builder).downcast())
                        }
                    }

                    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<16, T>>
                    where
                        T: ::planus::WriteAsOffset<self::FixedSizeList>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<Type>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }
                    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<17, T>>
                    where
                        T: ::planus::WriteAsOffset<self::Map>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<Type> {
                            ::planus::UnionOffset::new(17, (self.0).0.prepare(builder).downcast())
                        }
                    }

                    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<17, T>>
                    where
                        T: ::planus::WriteAsOffset<self::Map>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<Type>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }
                    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<18, T>>
                    where
                        T: ::planus::WriteAsOffset<self::Duration>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<Type> {
                            ::planus::UnionOffset::new(18, (self.0).0.prepare(builder).downcast())
                        }
                    }

                    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<18, T>>
                    where
                        T: ::planus::WriteAsOffset<self::Duration>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<Type>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }
                    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<19, T>>
                    where
                        T: ::planus::WriteAsOffset<self::LargeBinary>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<Type> {
                            ::planus::UnionOffset::new(19, (self.0).0.prepare(builder).downcast())
                        }
                    }

                    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<19, T>>
                    where
                        T: ::planus::WriteAsOffset<self::LargeBinary>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<Type>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }
                    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<20, T>>
                    where
                        T: ::planus::WriteAsOffset<self::LargeUtf8>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<Type> {
                            ::planus::UnionOffset::new(20, (self.0).0.prepare(builder).downcast())
                        }
                    }

                    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<20, T>>
                    where
                        T: ::planus::WriteAsOffset<self::LargeUtf8>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<Type>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }
                    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<21, T>>
                    where
                        T: ::planus::WriteAsOffset<self::LargeList>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<Type> {
                            ::planus::UnionOffset::new(21, (self.0).0.prepare(builder).downcast())
                        }
                    }

                    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<21, T>>
                    where
                        T: ::planus::WriteAsOffset<self::LargeList>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<Type>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }
                    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<22, T>>
                    where
                        T: ::planus::WriteAsOffset<self::RunEndEncoded>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<Type> {
                            ::planus::UnionOffset::new(22, (self.0).0.prepare(builder).downcast())
                        }
                    }

                    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<22, T>>
                    where
                        T: ::planus::WriteAsOffset<self::RunEndEncoded>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<Type>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }
                    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<23, T>>
                    where
                        T: ::planus::WriteAsOffset<self::BinaryView>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<Type> {
                            ::planus::UnionOffset::new(23, (self.0).0.prepare(builder).downcast())
                        }
                    }

                    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<23, T>>
                    where
                        T: ::planus::WriteAsOffset<self::BinaryView>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<Type>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }
                    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<24, T>>
                    where
                        T: ::planus::WriteAsOffset<self::Utf8View>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<Type> {
                            ::planus::UnionOffset::new(24, (self.0).0.prepare(builder).downcast())
                        }
                    }

                    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<24, T>>
                    where
                        T: ::planus::WriteAsOffset<self::Utf8View>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<Type>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }
                    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<25, T>>
                    where
                        T: ::planus::WriteAsOffset<self::ListView>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<Type> {
                            ::planus::UnionOffset::new(25, (self.0).0.prepare(builder).downcast())
                        }
                    }

                    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<25, T>>
                    where
                        T: ::planus::WriteAsOffset<self::ListView>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<Type>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }
                    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<26, T>>
                    where
                        T: ::planus::WriteAsOffset<self::LargeListView>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<Type> {
                            ::planus::UnionOffset::new(26, (self.0).0.prepare(builder).downcast())
                        }
                    }

                    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<26, T>>
                    where
                        T: ::planus::WriteAsOffset<self::LargeListView>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<Type>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }

                    /// Reference to a deserialized [Type].
                    #[derive(Copy, Clone, Debug)]
                    pub enum TypeRef<'a> {
                        Null(self::NullRef<'a>),
                        Int(self::IntRef<'a>),
                        FloatingPoint(self::FloatingPointRef<'a>),
                        Binary(self::BinaryRef<'a>),
                        Utf8(self::Utf8Ref<'a>),
                        Bool(self::BoolRef<'a>),
                        Decimal(self::DecimalRef<'a>),
                        Date(self::DateRef<'a>),
                        Time(self::TimeRef<'a>),
                        Timestamp(self::TimestampRef<'a>),
                        Interval(self::IntervalRef<'a>),
                        List(self::ListRef<'a>),
                        Struct(self::StructRef<'a>),
                        Union(self::UnionRef<'a>),
                        FixedSizeBinary(self::FixedSizeBinaryRef<'a>),
                        FixedSizeList(self::FixedSizeListRef<'a>),
                        Map(self::MapRef<'a>),
                        Duration(self::DurationRef<'a>),
                        LargeBinary(self::LargeBinaryRef<'a>),
                        LargeUtf8(self::LargeUtf8Ref<'a>),
                        LargeList(self::LargeListRef<'a>),
                        RunEndEncoded(self::RunEndEncodedRef<'a>),
                        BinaryView(self::BinaryViewRef<'a>),
                        Utf8View(self::Utf8ViewRef<'a>),
                        ListView(self::ListViewRef<'a>),
                        LargeListView(self::LargeListViewRef<'a>),
                    }

                    impl<'a> ::core::convert::TryFrom<TypeRef<'a>> for Type {
                        type Error = ::planus::Error;

                        fn try_from(value: TypeRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(match value {
                                TypeRef::Null(value) => {
                                    Self::Null(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::Int(value) => Self::Int(::planus::alloc::boxed::Box::new(
                                    ::core::convert::TryFrom::try_from(value)?,
                                )),

                                TypeRef::FloatingPoint(value) => {
                                    Self::FloatingPoint(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::Binary(value) => {
                                    Self::Binary(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::Utf8(value) => {
                                    Self::Utf8(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::Bool(value) => {
                                    Self::Bool(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::Decimal(value) => {
                                    Self::Decimal(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::Date(value) => {
                                    Self::Date(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::Time(value) => {
                                    Self::Time(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::Timestamp(value) => {
                                    Self::Timestamp(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::Interval(value) => {
                                    Self::Interval(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::List(value) => {
                                    Self::List(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::Struct(value) => {
                                    Self::Struct(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::Union(value) => {
                                    Self::Union(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::FixedSizeBinary(value) => {
                                    Self::FixedSizeBinary(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::FixedSizeList(value) => {
                                    Self::FixedSizeList(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::Map(value) => Self::Map(::planus::alloc::boxed::Box::new(
                                    ::core::convert::TryFrom::try_from(value)?,
                                )),

                                TypeRef::Duration(value) => {
                                    Self::Duration(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::LargeBinary(value) => {
                                    Self::LargeBinary(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::LargeUtf8(value) => {
                                    Self::LargeUtf8(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::LargeList(value) => {
                                    Self::LargeList(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::RunEndEncoded(value) => {
                                    Self::RunEndEncoded(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::BinaryView(value) => {
                                    Self::BinaryView(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::Utf8View(value) => {
                                    Self::Utf8View(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::ListView(value) => {
                                    Self::ListView(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::LargeListView(value) => {
                                    Self::LargeListView(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }
                            })
                        }
                    }

                    impl<'a> ::planus::TableReadUnion<'a> for TypeRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            tag: u8,
                            field_offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            match tag {
                                1 => ::core::result::Result::Ok(Self::Null(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                2 => ::core::result::Result::Ok(Self::Int(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                3 => ::core::result::Result::Ok(Self::FloatingPoint(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                4 => ::core::result::Result::Ok(Self::Binary(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                5 => ::core::result::Result::Ok(Self::Utf8(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                6 => ::core::result::Result::Ok(Self::Bool(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                7 => ::core::result::Result::Ok(Self::Decimal(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                8 => ::core::result::Result::Ok(Self::Date(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                9 => ::core::result::Result::Ok(Self::Time(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                10 => ::core::result::Result::Ok(Self::Timestamp(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                11 => ::core::result::Result::Ok(Self::Interval(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                12 => ::core::result::Result::Ok(Self::List(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                13 => ::core::result::Result::Ok(Self::Struct(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                14 => ::core::result::Result::Ok(Self::Union(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                15 => ::core::result::Result::Ok(Self::FixedSizeBinary(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                16 => ::core::result::Result::Ok(Self::FixedSizeList(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                17 => ::core::result::Result::Ok(Self::Map(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                18 => ::core::result::Result::Ok(Self::Duration(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                19 => ::core::result::Result::Ok(Self::LargeBinary(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                20 => ::core::result::Result::Ok(Self::LargeUtf8(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                21 => ::core::result::Result::Ok(Self::LargeList(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                22 => ::core::result::Result::Ok(Self::RunEndEncoded(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                23 => ::core::result::Result::Ok(Self::BinaryView(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                24 => ::core::result::Result::Ok(Self::Utf8View(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                25 => ::core::result::Result::Ok(Self::ListView(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                26 => ::core::result::Result::Ok(Self::LargeListView(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                _ => ::core::result::Result::Err(
                                    ::planus::errors::ErrorKind::UnknownUnionTag { tag },
                                ),
                            }
                        }
                    }

                    impl<'a> ::planus::VectorReadUnion<'a> for TypeRef<'a> {
                        const VECTOR_NAME: &'static str = "[TypeRef]";
                    }

                    ///  ----------------------------------------------------------------------
                    ///  user defined key value pairs to add custom metadata to arrow
                    ///  key namespacing is the responsibility of the user
                    ///
                    /// Generated from these locations:
                    /// * Table `KeyValue` in the file `Schema.fbs:475`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct KeyValue {
                        /// The field `key` in the table `KeyValue`
                        pub key: ::core::option::Option<::planus::alloc::string::String>,
                        /// The field `value` in the table `KeyValue`
                        pub value: ::core::option::Option<::planus::alloc::string::String>,
                    }

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for KeyValue {
                        fn default() -> Self {
                            Self {
                                key: ::core::default::Default::default(),
                                value: ::core::default::Default::default(),
                            }
                        }
                    }

                    impl KeyValue {
                        /// Creates a [KeyValueBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> KeyValueBuilder<()> {
                            KeyValueBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            field_key: impl ::planus::WriteAsOptional<
                                ::planus::Offset<::core::primitive::str>,
                            >,
                            field_value: impl ::planus::WriteAsOptional<
                                ::planus::Offset<::core::primitive::str>,
                            >,
                        ) -> ::planus::Offset<Self> {
                            let prepared_key = field_key.prepare(builder);
                            let prepared_value = field_value.prepare(builder);

                            let mut table_writer: ::planus::table_writer::TableWriter<8> =
                                ::core::default::Default::default();
                            if prepared_key.is_some() {
                                table_writer.write_entry::<::planus::Offset<str>>(0);
                            }
                            if prepared_value.is_some() {
                                table_writer.write_entry::<::planus::Offset<str>>(1);
                            }

                            unsafe {
                                table_writer.finish(builder, |object_writer| {
                                    if let ::core::option::Option::Some(prepared_key) = prepared_key
                                    {
                                        object_writer.write::<_, _, 4>(&prepared_key);
                                    }
                                    if let ::core::option::Option::Some(prepared_value) =
                                        prepared_value
                                    {
                                        object_writer.write::<_, _, 4>(&prepared_value);
                                    }
                                });
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<KeyValue>> for KeyValue {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<KeyValue> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<KeyValue>> for KeyValue {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<KeyValue>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<KeyValue> for KeyValue {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<KeyValue> {
                            KeyValue::create(builder, &self.key, &self.value)
                        }
                    }

                    /// Builder for serializing an instance of the [KeyValue] type.
                    ///
                    /// Can be created using the [KeyValue::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct KeyValueBuilder<State>(State);

                    impl KeyValueBuilder<()> {
                        /// Setter for the [`key` field](KeyValue#structfield.key).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn key<T0>(self, value: T0) -> KeyValueBuilder<(T0,)>
                        where
                            T0: ::planus::WriteAsOptional<::planus::Offset<::core::primitive::str>>,
                        {
                            KeyValueBuilder((value,))
                        }

                        /// Sets the [`key` field](KeyValue#structfield.key) to null.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn key_as_null(self) -> KeyValueBuilder<((),)> {
                            self.key(())
                        }
                    }

                    impl<T0> KeyValueBuilder<(T0,)> {
                        /// Setter for the [`value` field](KeyValue#structfield.value).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn value<T1>(self, value: T1) -> KeyValueBuilder<(T0, T1)>
                        where
                            T1: ::planus::WriteAsOptional<::planus::Offset<::core::primitive::str>>,
                        {
                            let (v0,) = self.0;
                            KeyValueBuilder((v0, value))
                        }

                        /// Sets the [`value` field](KeyValue#structfield.value) to null.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn value_as_null(self) -> KeyValueBuilder<(T0, ())> {
                            self.value(())
                        }
                    }

                    impl<T0, T1> KeyValueBuilder<(T0, T1)> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [KeyValue].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<KeyValue>
                        where
                            Self: ::planus::WriteAsOffset<KeyValue>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsOptional<::planus::Offset<::core::primitive::str>>,
                            T1: ::planus::WriteAsOptional<::planus::Offset<::core::primitive::str>>,
                        > ::planus::WriteAs<::planus::Offset<KeyValue>>
                        for KeyValueBuilder<(T0, T1)>
                    {
                        type Prepared = ::planus::Offset<KeyValue>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<KeyValue> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsOptional<::planus::Offset<::core::primitive::str>>,
                            T1: ::planus::WriteAsOptional<::planus::Offset<::core::primitive::str>>,
                        > ::planus::WriteAsOptional<::planus::Offset<KeyValue>>
                        for KeyValueBuilder<(T0, T1)>
                    {
                        type Prepared = ::planus::Offset<KeyValue>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<KeyValue>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsOptional<::planus::Offset<::core::primitive::str>>,
                            T1: ::planus::WriteAsOptional<::planus::Offset<::core::primitive::str>>,
                        > ::planus::WriteAsOffset<KeyValue> for KeyValueBuilder<(T0, T1)>
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<KeyValue> {
                            let (v0, v1) = &self.0;
                            KeyValue::create(builder, v0, v1)
                        }
                    }

                    /// Reference to a deserialized [KeyValue].
                    #[derive(Copy, Clone)]
                    pub struct KeyValueRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> KeyValueRef<'a> {
                        /// Getter for the [`key` field](KeyValue#structfield.key).
                        #[inline]
                        pub fn key(
                            &self,
                        ) -> ::planus::Result<::core::option::Option<&'a ::core::primitive::str>>
                        {
                            self.0.access(0, "KeyValue", "key")
                        }

                        /// Getter for the [`value` field](KeyValue#structfield.value).
                        #[inline]
                        pub fn value(
                            &self,
                        ) -> ::planus::Result<::core::option::Option<&'a ::core::primitive::str>>
                        {
                            self.0.access(1, "KeyValue", "value")
                        }
                    }

                    impl<'a> ::core::fmt::Debug for KeyValueRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("KeyValueRef");
                            if let ::core::option::Option::Some(field_key) = self.key().transpose()
                            {
                                f.field("key", &field_key);
                            }
                            if let ::core::option::Option::Some(field_value) =
                                self.value().transpose()
                            {
                                f.field("value", &field_value);
                            }
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<KeyValueRef<'a>> for KeyValue {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: KeyValueRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                key: value.key()?.map(::core::convert::Into::into),
                                value: value.value()?.map(::core::convert::Into::into),
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for KeyValueRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for KeyValueRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[KeyValueRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<KeyValue>> for KeyValue {
                        type Value = ::planus::Offset<KeyValue>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<KeyValue>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for KeyValueRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[KeyValueRef]", "read_as_root", 0)
                            })
                        }
                    }

                    ///  ----------------------------------------------------------------------
                    ///  Dictionary encoding metadata
                    ///  Maintained for forwards compatibility, in the future
                    ///  Dictionaries might be explicit maps between integers and values
                    ///  allowing for non-contiguous index values
                    ///
                    /// Generated from these locations:
                    /// * Enum `DictionaryKind` in the file `Schema.fbs:485`
                    #[derive(
                        Copy,
                        Clone,
                        Debug,
                        PartialEq,
                        Eq,
                        PartialOrd,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    #[repr(i16)]
                    pub enum DictionaryKind {
                        /// The variant `DenseArray` in the enum `DictionaryKind`
                        DenseArray = 0,
                    }

                    impl DictionaryKind {
                        /// Array containing all valid variants of DictionaryKind
                        pub const ENUM_VALUES: [Self; 1] = [Self::DenseArray];
                    }

                    impl ::core::convert::TryFrom<i16> for DictionaryKind {
                        type Error = ::planus::errors::UnknownEnumTagKind;
                        #[inline]
                        fn try_from(
                            value: i16,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTagKind>
                        {
                            #[allow(clippy::match_single_binding)]
                            match value {
                                0 => ::core::result::Result::Ok(DictionaryKind::DenseArray),

                                _ => ::core::result::Result::Err(
                                    ::planus::errors::UnknownEnumTagKind { tag: value as i128 },
                                ),
                            }
                        }
                    }

                    impl ::core::convert::From<DictionaryKind> for i16 {
                        #[inline]
                        fn from(value: DictionaryKind) -> Self {
                            value as i16
                        }
                    }

                    /// # Safety
                    /// The Planus compiler correctly calculates `ALIGNMENT` and `SIZE`.
                    unsafe impl ::planus::Primitive for DictionaryKind {
                        const ALIGNMENT: usize = 2;
                        const SIZE: usize = 2;
                    }

                    impl ::planus::WriteAsPrimitive<DictionaryKind> for DictionaryKind {
                        #[inline]
                        fn write<const N: usize>(
                            &self,
                            cursor: ::planus::Cursor<'_, N>,
                            buffer_position: u32,
                        ) {
                            (*self as i16).write(cursor, buffer_position);
                        }
                    }

                    impl ::planus::WriteAs<DictionaryKind> for DictionaryKind {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> DictionaryKind {
                            *self
                        }
                    }

                    impl ::planus::WriteAsDefault<DictionaryKind, DictionaryKind> for DictionaryKind {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                            default: &DictionaryKind,
                        ) -> ::core::option::Option<DictionaryKind> {
                            if self == default {
                                ::core::option::Option::None
                            } else {
                                ::core::option::Option::Some(*self)
                            }
                        }
                    }

                    impl ::planus::WriteAsOptional<DictionaryKind> for DictionaryKind {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<DictionaryKind> {
                            ::core::option::Option::Some(*self)
                        }
                    }

                    impl<'buf> ::planus::TableRead<'buf> for DictionaryKind {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            let n: i16 = ::planus::TableRead::from_buffer(buffer, offset)?;
                            ::core::result::Result::Ok(::core::convert::TryInto::try_into(n)?)
                        }
                    }

                    impl<'buf> ::planus::VectorReadInner<'buf> for DictionaryKind {
                        type Error = ::planus::errors::UnknownEnumTag;
                        const STRIDE: usize = 2;
                        #[inline]
                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTag>
                        {
                            let value = unsafe {
                                <i16 as ::planus::VectorRead>::from_buffer(buffer, offset)
                            };
                            let value: ::core::result::Result<Self, _> =
                                ::core::convert::TryInto::try_into(value);
                            value.map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "DictionaryKind",
                                    "VectorRead::from_buffer",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<DictionaryKind> for DictionaryKind {
                        const STRIDE: usize = 2;

                        type Value = Self;

                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self {
                            *self
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[Self],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 2];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (2 * i) as u32,
                                );
                            }
                        }
                    }

                    /// The table `DictionaryEncoding` in the namespace `org.apache.arrow.flatbuf`
                    ///
                    /// Generated from these locations:
                    /// * Table `DictionaryEncoding` in the file `Schema.fbs:486`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct DictionaryEncoding {
                        ///  The known dictionary id in the application where this data is used. In
                        ///  the file or streaming formats, the dictionary ids are found in the
                        ///  DictionaryBatch messages
                        pub id: i64,
                        ///  The dictionary indices are constrained to be non-negative integers. If
                        ///  this field is null, the indices must be signed int32. To maximize
                        ///  cross-language compatibility and performance, implementations are
                        ///  recommended to prefer signed integer types over unsigned integer types
                        ///  and to avoid uint64 indices unless they are required by an application.
                        pub index_type:
                            ::core::option::Option<::planus::alloc::boxed::Box<self::Int>>,
                        ///  By default, dictionaries are not ordered, or the order does not have
                        ///  semantic meaning. In some statistical, applications, dictionary-encoding
                        ///  is used to represent ordered categorical data, and we provide a way to
                        ///  preserve that metadata here
                        pub is_ordered: bool,
                        /// The field `dictionaryKind` in the table `DictionaryEncoding`
                        pub dictionary_kind: self::DictionaryKind,
                    }

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for DictionaryEncoding {
                        fn default() -> Self {
                            Self {
                                id: 0,
                                index_type: ::core::default::Default::default(),
                                is_ordered: false,
                                dictionary_kind: self::DictionaryKind::DenseArray,
                            }
                        }
                    }

                    impl DictionaryEncoding {
                        /// Creates a [DictionaryEncodingBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> DictionaryEncodingBuilder<()> {
                            DictionaryEncodingBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            field_id: impl ::planus::WriteAsDefault<i64, i64>,
                            field_index_type: impl ::planus::WriteAsOptional<
                                ::planus::Offset<self::Int>,
                            >,
                            field_is_ordered: impl ::planus::WriteAsDefault<bool, bool>,
                            field_dictionary_kind: impl ::planus::WriteAsDefault<
                                self::DictionaryKind,
                                self::DictionaryKind,
                            >,
                        ) -> ::planus::Offset<Self> {
                            let prepared_id = field_id.prepare(builder, &0);
                            let prepared_index_type = field_index_type.prepare(builder);
                            let prepared_is_ordered = field_is_ordered.prepare(builder, &false);
                            let prepared_dictionary_kind = field_dictionary_kind
                                .prepare(builder, &self::DictionaryKind::DenseArray);

                            let mut table_writer: ::planus::table_writer::TableWriter<12> =
                                ::core::default::Default::default();
                            if prepared_id.is_some() {
                                table_writer.write_entry::<i64>(0);
                            }
                            if prepared_index_type.is_some() {
                                table_writer.write_entry::<::planus::Offset<self::Int>>(1);
                            }
                            if prepared_dictionary_kind.is_some() {
                                table_writer.write_entry::<self::DictionaryKind>(3);
                            }
                            if prepared_is_ordered.is_some() {
                                table_writer.write_entry::<bool>(2);
                            }

                            unsafe {
                                table_writer.finish(builder, |object_writer| {
                                    if let ::core::option::Option::Some(prepared_id) = prepared_id {
                                        object_writer.write::<_, _, 8>(&prepared_id);
                                    }
                                    if let ::core::option::Option::Some(prepared_index_type) =
                                        prepared_index_type
                                    {
                                        object_writer.write::<_, _, 4>(&prepared_index_type);
                                    }
                                    if let ::core::option::Option::Some(prepared_dictionary_kind) =
                                        prepared_dictionary_kind
                                    {
                                        object_writer.write::<_, _, 2>(&prepared_dictionary_kind);
                                    }
                                    if let ::core::option::Option::Some(prepared_is_ordered) =
                                        prepared_is_ordered
                                    {
                                        object_writer.write::<_, _, 1>(&prepared_is_ordered);
                                    }
                                });
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<DictionaryEncoding>> for DictionaryEncoding {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<DictionaryEncoding> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<DictionaryEncoding>> for DictionaryEncoding {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<DictionaryEncoding>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<DictionaryEncoding> for DictionaryEncoding {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<DictionaryEncoding> {
                            DictionaryEncoding::create(
                                builder,
                                self.id,
                                &self.index_type,
                                self.is_ordered,
                                self.dictionary_kind,
                            )
                        }
                    }

                    /// Builder for serializing an instance of the [DictionaryEncoding] type.
                    ///
                    /// Can be created using the [DictionaryEncoding::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct DictionaryEncodingBuilder<State>(State);

                    impl DictionaryEncodingBuilder<()> {
                        /// Setter for the [`id` field](DictionaryEncoding#structfield.id).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn id<T0>(self, value: T0) -> DictionaryEncodingBuilder<(T0,)>
                        where
                            T0: ::planus::WriteAsDefault<i64, i64>,
                        {
                            DictionaryEncodingBuilder((value,))
                        }

                        /// Sets the [`id` field](DictionaryEncoding#structfield.id) to the default value.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn id_as_default(
                            self,
                        ) -> DictionaryEncodingBuilder<(::planus::DefaultValue,)>
                        {
                            self.id(::planus::DefaultValue)
                        }
                    }

                    impl<T0> DictionaryEncodingBuilder<(T0,)> {
                        /// Setter for the [`indexType` field](DictionaryEncoding#structfield.index_type).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn index_type<T1>(
                            self,
                            value: T1,
                        ) -> DictionaryEncodingBuilder<(T0, T1)>
                        where
                            T1: ::planus::WriteAsOptional<::planus::Offset<self::Int>>,
                        {
                            let (v0,) = self.0;
                            DictionaryEncodingBuilder((v0, value))
                        }

                        /// Sets the [`indexType` field](DictionaryEncoding#structfield.index_type) to null.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn index_type_as_null(self) -> DictionaryEncodingBuilder<(T0, ())> {
                            self.index_type(())
                        }
                    }

                    impl<T0, T1> DictionaryEncodingBuilder<(T0, T1)> {
                        /// Setter for the [`isOrdered` field](DictionaryEncoding#structfield.is_ordered).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn is_ordered<T2>(
                            self,
                            value: T2,
                        ) -> DictionaryEncodingBuilder<(T0, T1, T2)>
                        where
                            T2: ::planus::WriteAsDefault<bool, bool>,
                        {
                            let (v0, v1) = self.0;
                            DictionaryEncodingBuilder((v0, v1, value))
                        }

                        /// Sets the [`isOrdered` field](DictionaryEncoding#structfield.is_ordered) to the default value.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn is_ordered_as_default(
                            self,
                        ) -> DictionaryEncodingBuilder<(T0, T1, ::planus::DefaultValue)>
                        {
                            self.is_ordered(::planus::DefaultValue)
                        }
                    }

                    impl<T0, T1, T2> DictionaryEncodingBuilder<(T0, T1, T2)> {
                        /// Setter for the [`dictionaryKind` field](DictionaryEncoding#structfield.dictionary_kind).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn dictionary_kind<T3>(
                            self,
                            value: T3,
                        ) -> DictionaryEncodingBuilder<(T0, T1, T2, T3)>
                        where
                            T3: ::planus::WriteAsDefault<
                                self::DictionaryKind,
                                self::DictionaryKind,
                            >,
                        {
                            let (v0, v1, v2) = self.0;
                            DictionaryEncodingBuilder((v0, v1, v2, value))
                        }

                        /// Sets the [`dictionaryKind` field](DictionaryEncoding#structfield.dictionary_kind) to the default value.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn dictionary_kind_as_default(
                            self,
                        ) -> DictionaryEncodingBuilder<(T0, T1, T2, ::planus::DefaultValue)>
                        {
                            self.dictionary_kind(::planus::DefaultValue)
                        }
                    }

                    impl<T0, T1, T2, T3> DictionaryEncodingBuilder<(T0, T1, T2, T3)> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [DictionaryEncoding].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<DictionaryEncoding>
                        where
                            Self: ::planus::WriteAsOffset<DictionaryEncoding>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<i64, i64>,
                            T1: ::planus::WriteAsOptional<::planus::Offset<self::Int>>,
                            T2: ::planus::WriteAsDefault<bool, bool>,
                            T3: ::planus::WriteAsDefault<self::DictionaryKind, self::DictionaryKind>,
                        >
                        ::planus::WriteAs<::planus::Offset<DictionaryEncoding>>
                        for DictionaryEncodingBuilder<(T0, T1, T2, T3)>
                    {
                        type Prepared = ::planus::Offset<DictionaryEncoding>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<DictionaryEncoding> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<i64, i64>,
                            T1: ::planus::WriteAsOptional<::planus::Offset<self::Int>>,
                            T2: ::planus::WriteAsDefault<bool, bool>,
                            T3: ::planus::WriteAsDefault<self::DictionaryKind, self::DictionaryKind>,
                        >
                        ::planus::WriteAsOptional<::planus::Offset<DictionaryEncoding>>
                        for DictionaryEncodingBuilder<(T0, T1, T2, T3)>
                    {
                        type Prepared = ::planus::Offset<DictionaryEncoding>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<DictionaryEncoding>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<i64, i64>,
                            T1: ::planus::WriteAsOptional<::planus::Offset<self::Int>>,
                            T2: ::planus::WriteAsDefault<bool, bool>,
                            T3: ::planus::WriteAsDefault<self::DictionaryKind, self::DictionaryKind>,
                        > ::planus::WriteAsOffset<DictionaryEncoding>
                        for DictionaryEncodingBuilder<(T0, T1, T2, T3)>
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<DictionaryEncoding> {
                            let (v0, v1, v2, v3) = &self.0;
                            DictionaryEncoding::create(builder, v0, v1, v2, v3)
                        }
                    }

                    /// Reference to a deserialized [DictionaryEncoding].
                    #[derive(Copy, Clone)]
                    pub struct DictionaryEncodingRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> DictionaryEncodingRef<'a> {
                        /// Getter for the [`id` field](DictionaryEncoding#structfield.id).
                        #[inline]
                        pub fn id(&self) -> ::planus::Result<i64> {
                            ::core::result::Result::Ok(
                                self.0.access(0, "DictionaryEncoding", "id")?.unwrap_or(0),
                            )
                        }

                        /// Getter for the [`indexType` field](DictionaryEncoding#structfield.index_type).
                        #[inline]
                        pub fn index_type(
                            &self,
                        ) -> ::planus::Result<::core::option::Option<self::IntRef<'a>>>
                        {
                            self.0.access(1, "DictionaryEncoding", "index_type")
                        }

                        /// Getter for the [`isOrdered` field](DictionaryEncoding#structfield.is_ordered).
                        #[inline]
                        pub fn is_ordered(&self) -> ::planus::Result<bool> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(2, "DictionaryEncoding", "is_ordered")?
                                    .unwrap_or(false),
                            )
                        }

                        /// Getter for the [`dictionaryKind` field](DictionaryEncoding#structfield.dictionary_kind).
                        #[inline]
                        pub fn dictionary_kind(&self) -> ::planus::Result<self::DictionaryKind> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(3, "DictionaryEncoding", "dictionary_kind")?
                                    .unwrap_or(self::DictionaryKind::DenseArray),
                            )
                        }
                    }

                    impl<'a> ::core::fmt::Debug for DictionaryEncodingRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("DictionaryEncodingRef");
                            f.field("id", &self.id());
                            if let ::core::option::Option::Some(field_index_type) =
                                self.index_type().transpose()
                            {
                                f.field("index_type", &field_index_type);
                            }
                            f.field("is_ordered", &self.is_ordered());
                            f.field("dictionary_kind", &self.dictionary_kind());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<DictionaryEncodingRef<'a>> for DictionaryEncoding {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: DictionaryEncodingRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                id: ::core::convert::TryInto::try_into(value.id()?)?,
                                index_type: if let ::core::option::Option::Some(index_type) =
                                    value.index_type()?
                                {
                                    ::core::option::Option::Some(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryInto::try_into(index_type)?,
                                    ))
                                } else {
                                    ::core::option::Option::None
                                },
                                is_ordered: ::core::convert::TryInto::try_into(
                                    value.is_ordered()?,
                                )?,
                                dictionary_kind: ::core::convert::TryInto::try_into(
                                    value.dictionary_kind()?,
                                )?,
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for DictionaryEncodingRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for DictionaryEncodingRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[DictionaryEncodingRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<DictionaryEncoding>> for DictionaryEncoding {
                        type Value = ::planus::Offset<DictionaryEncoding>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<DictionaryEncoding>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for DictionaryEncodingRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[DictionaryEncodingRef]",
                                    "read_as_root",
                                    0,
                                )
                            })
                        }
                    }

                    ///  ----------------------------------------------------------------------
                    ///  A field represents a named column in a record / row batch or child of a
                    ///  nested type.
                    ///
                    /// Generated from these locations:
                    /// * Table `Field` in the file `Schema.fbs:512`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct Field {
                        ///  Name is not required (e.g., in a List)
                        pub name: ::core::option::Option<::planus::alloc::string::String>,
                        ///  Whether or not this field can contain nulls. Should be true in general.
                        pub nullable: bool,
                        ///  This is the type of the decoded value if the field is dictionary encoded.
                        pub type_: ::core::option::Option<self::Type>,
                        ///  Present only if the field is dictionary encoded.
                        pub dictionary: ::core::option::Option<
                            ::planus::alloc::boxed::Box<self::DictionaryEncoding>,
                        >,
                        ///  children apply only to nested data types like Struct, List and Union. For
                        ///  primitive types children will have length 0.
                        pub children:
                            ::core::option::Option<::planus::alloc::vec::Vec<self::Field>>,
                        ///  User-defined metadata
                        pub custom_metadata:
                            ::core::option::Option<::planus::alloc::vec::Vec<self::KeyValue>>,
                    }

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for Field {
                        fn default() -> Self {
                            Self {
                                name: ::core::default::Default::default(),
                                nullable: false,
                                type_: ::core::default::Default::default(),
                                dictionary: ::core::default::Default::default(),
                                children: ::core::default::Default::default(),
                                custom_metadata: ::core::default::Default::default(),
                            }
                        }
                    }

                    impl Field {
                        /// Creates a [FieldBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> FieldBuilder<()> {
                            FieldBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            field_name: impl ::planus::WriteAsOptional<
                                ::planus::Offset<::core::primitive::str>,
                            >,
                            field_nullable: impl ::planus::WriteAsDefault<bool, bool>,
                            field_type_: impl ::planus::WriteAsOptionalUnion<self::Type>,
                            field_dictionary: impl ::planus::WriteAsOptional<
                                ::planus::Offset<self::DictionaryEncoding>,
                            >,
                            field_children: impl ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::Field>]>,
                            >,
                            field_custom_metadata: impl ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::KeyValue>]>,
                            >,
                        ) -> ::planus::Offset<Self> {
                            let prepared_name = field_name.prepare(builder);
                            let prepared_nullable = field_nullable.prepare(builder, &false);
                            let prepared_type_ = field_type_.prepare(builder);
                            let prepared_dictionary = field_dictionary.prepare(builder);
                            let prepared_children = field_children.prepare(builder);
                            let prepared_custom_metadata = field_custom_metadata.prepare(builder);

                            let mut table_writer: ::planus::table_writer::TableWriter<18> =
                                ::core::default::Default::default();
                            if prepared_name.is_some() {
                                table_writer.write_entry::<::planus::Offset<str>>(0);
                            }
                            if prepared_type_.is_some() {
                                table_writer.write_entry::<::planus::Offset<self::Type>>(3);
                            }
                            if prepared_dictionary.is_some() {
                                table_writer
                                    .write_entry::<::planus::Offset<self::DictionaryEncoding>>(4);
                            }
                            if prepared_children.is_some() {
                                table_writer.write_entry::<::planus::Offset<[::planus::Offset<self::Field>]>>(5);
                            }
                            if prepared_custom_metadata.is_some() {
                                table_writer.write_entry::<::planus::Offset<[::planus::Offset<self::KeyValue>]>>(6);
                            }
                            if prepared_nullable.is_some() {
                                table_writer.write_entry::<bool>(1);
                            }
                            if prepared_type_.is_some() {
                                table_writer.write_entry::<u8>(2);
                            }

                            unsafe {
                                table_writer.finish(builder, |object_writer| {
                                    if let ::core::option::Option::Some(prepared_name) =
                                        prepared_name
                                    {
                                        object_writer.write::<_, _, 4>(&prepared_name);
                                    }
                                    if let ::core::option::Option::Some(prepared_type_) =
                                        prepared_type_
                                    {
                                        object_writer.write::<_, _, 4>(&prepared_type_.offset());
                                    }
                                    if let ::core::option::Option::Some(prepared_dictionary) =
                                        prepared_dictionary
                                    {
                                        object_writer.write::<_, _, 4>(&prepared_dictionary);
                                    }
                                    if let ::core::option::Option::Some(prepared_children) =
                                        prepared_children
                                    {
                                        object_writer.write::<_, _, 4>(&prepared_children);
                                    }
                                    if let ::core::option::Option::Some(prepared_custom_metadata) =
                                        prepared_custom_metadata
                                    {
                                        object_writer.write::<_, _, 4>(&prepared_custom_metadata);
                                    }
                                    if let ::core::option::Option::Some(prepared_nullable) =
                                        prepared_nullable
                                    {
                                        object_writer.write::<_, _, 1>(&prepared_nullable);
                                    }
                                    if let ::core::option::Option::Some(prepared_type_) =
                                        prepared_type_
                                    {
                                        object_writer.write::<_, _, 1>(&prepared_type_.tag());
                                    }
                                });
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Field>> for Field {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Field> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Field>> for Field {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Field>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Field> for Field {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Field> {
                            Field::create(
                                builder,
                                &self.name,
                                self.nullable,
                                &self.type_,
                                &self.dictionary,
                                &self.children,
                                &self.custom_metadata,
                            )
                        }
                    }

                    /// Builder for serializing an instance of the [Field] type.
                    ///
                    /// Can be created using the [Field::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct FieldBuilder<State>(State);

                    impl FieldBuilder<()> {
                        /// Setter for the [`name` field](Field#structfield.name).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn name<T0>(self, value: T0) -> FieldBuilder<(T0,)>
                        where
                            T0: ::planus::WriteAsOptional<::planus::Offset<::core::primitive::str>>,
                        {
                            FieldBuilder((value,))
                        }

                        /// Sets the [`name` field](Field#structfield.name) to null.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn name_as_null(self) -> FieldBuilder<((),)> {
                            self.name(())
                        }
                    }

                    impl<T0> FieldBuilder<(T0,)> {
                        /// Setter for the [`nullable` field](Field#structfield.nullable).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn nullable<T1>(self, value: T1) -> FieldBuilder<(T0, T1)>
                        where
                            T1: ::planus::WriteAsDefault<bool, bool>,
                        {
                            let (v0,) = self.0;
                            FieldBuilder((v0, value))
                        }

                        /// Sets the [`nullable` field](Field#structfield.nullable) to the default value.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn nullable_as_default(
                            self,
                        ) -> FieldBuilder<(T0, ::planus::DefaultValue)> {
                            self.nullable(::planus::DefaultValue)
                        }
                    }

                    impl<T0, T1> FieldBuilder<(T0, T1)> {
                        /// Setter for the [`type` field](Field#structfield.type_).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn type_<T2>(self, value: T2) -> FieldBuilder<(T0, T1, T2)>
                        where
                            T2: ::planus::WriteAsOptionalUnion<self::Type>,
                        {
                            let (v0, v1) = self.0;
                            FieldBuilder((v0, v1, value))
                        }

                        /// Sets the [`type` field](Field#structfield.type_) to null.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn type_as_null(self) -> FieldBuilder<(T0, T1, ())> {
                            self.type_(())
                        }
                    }

                    impl<T0, T1, T2> FieldBuilder<(T0, T1, T2)> {
                        /// Setter for the [`dictionary` field](Field#structfield.dictionary).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn dictionary<T3>(self, value: T3) -> FieldBuilder<(T0, T1, T2, T3)>
                        where
                            T3: ::planus::WriteAsOptional<
                                ::planus::Offset<self::DictionaryEncoding>,
                            >,
                        {
                            let (v0, v1, v2) = self.0;
                            FieldBuilder((v0, v1, v2, value))
                        }

                        /// Sets the [`dictionary` field](Field#structfield.dictionary) to null.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn dictionary_as_null(self) -> FieldBuilder<(T0, T1, T2, ())> {
                            self.dictionary(())
                        }
                    }

                    impl<T0, T1, T2, T3> FieldBuilder<(T0, T1, T2, T3)> {
                        /// Setter for the [`children` field](Field#structfield.children).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn children<T4>(self, value: T4) -> FieldBuilder<(T0, T1, T2, T3, T4)>
                        where
                            T4: ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::Field>]>,
                            >,
                        {
                            let (v0, v1, v2, v3) = self.0;
                            FieldBuilder((v0, v1, v2, v3, value))
                        }

                        /// Sets the [`children` field](Field#structfield.children) to null.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn children_as_null(self) -> FieldBuilder<(T0, T1, T2, T3, ())> {
                            self.children(())
                        }
                    }

                    impl<T0, T1, T2, T3, T4> FieldBuilder<(T0, T1, T2, T3, T4)> {
                        /// Setter for the [`custom_metadata` field](Field#structfield.custom_metadata).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn custom_metadata<T5>(
                            self,
                            value: T5,
                        ) -> FieldBuilder<(T0, T1, T2, T3, T4, T5)>
                        where
                            T5: ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::KeyValue>]>,
                            >,
                        {
                            let (v0, v1, v2, v3, v4) = self.0;
                            FieldBuilder((v0, v1, v2, v3, v4, value))
                        }

                        /// Sets the [`custom_metadata` field](Field#structfield.custom_metadata) to null.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn custom_metadata_as_null(
                            self,
                        ) -> FieldBuilder<(T0, T1, T2, T3, T4, ())> {
                            self.custom_metadata(())
                        }
                    }

                    impl<T0, T1, T2, T3, T4, T5> FieldBuilder<(T0, T1, T2, T3, T4, T5)> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Field].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Field>
                        where
                            Self: ::planus::WriteAsOffset<Field>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsOptional<::planus::Offset<::core::primitive::str>>,
                            T1: ::planus::WriteAsDefault<bool, bool>,
                            T2: ::planus::WriteAsOptionalUnion<self::Type>,
                            T3: ::planus::WriteAsOptional<::planus::Offset<self::DictionaryEncoding>>,
                            T4: ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::Field>]>,
                            >,
                            T5: ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::KeyValue>]>,
                            >,
                        > ::planus::WriteAs<::planus::Offset<Field>>
                        for FieldBuilder<(T0, T1, T2, T3, T4, T5)>
                    {
                        type Prepared = ::planus::Offset<Field>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Field> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsOptional<::planus::Offset<::core::primitive::str>>,
                            T1: ::planus::WriteAsDefault<bool, bool>,
                            T2: ::planus::WriteAsOptionalUnion<self::Type>,
                            T3: ::planus::WriteAsOptional<::planus::Offset<self::DictionaryEncoding>>,
                            T4: ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::Field>]>,
                            >,
                            T5: ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::KeyValue>]>,
                            >,
                        > ::planus::WriteAsOptional<::planus::Offset<Field>>
                        for FieldBuilder<(T0, T1, T2, T3, T4, T5)>
                    {
                        type Prepared = ::planus::Offset<Field>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Field>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsOptional<::planus::Offset<::core::primitive::str>>,
                            T1: ::planus::WriteAsDefault<bool, bool>,
                            T2: ::planus::WriteAsOptionalUnion<self::Type>,
                            T3: ::planus::WriteAsOptional<::planus::Offset<self::DictionaryEncoding>>,
                            T4: ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::Field>]>,
                            >,
                            T5: ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::KeyValue>]>,
                            >,
                        > ::planus::WriteAsOffset<Field>
                        for FieldBuilder<(T0, T1, T2, T3, T4, T5)>
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Field> {
                            let (v0, v1, v2, v3, v4, v5) = &self.0;
                            Field::create(builder, v0, v1, v2, v3, v4, v5)
                        }
                    }

                    /// Reference to a deserialized [Field].
                    #[derive(Copy, Clone)]
                    pub struct FieldRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> FieldRef<'a> {
                        /// Getter for the [`name` field](Field#structfield.name).
                        #[inline]
                        pub fn name(
                            &self,
                        ) -> ::planus::Result<::core::option::Option<&'a ::core::primitive::str>>
                        {
                            self.0.access(0, "Field", "name")
                        }

                        /// Getter for the [`nullable` field](Field#structfield.nullable).
                        #[inline]
                        pub fn nullable(&self) -> ::planus::Result<bool> {
                            ::core::result::Result::Ok(
                                self.0.access(1, "Field", "nullable")?.unwrap_or(false),
                            )
                        }

                        /// Getter for the [`type` field](Field#structfield.type_).
                        #[inline]
                        pub fn type_(
                            &self,
                        ) -> ::planus::Result<::core::option::Option<self::TypeRef<'a>>>
                        {
                            self.0.access_union(2, "Field", "type_")
                        }

                        /// Getter for the [`dictionary` field](Field#structfield.dictionary).
                        #[inline]
                        pub fn dictionary(
                            &self,
                        ) -> ::planus::Result<::core::option::Option<self::DictionaryEncodingRef<'a>>>
                        {
                            self.0.access(4, "Field", "dictionary")
                        }

                        /// Getter for the [`children` field](Field#structfield.children).
                        #[inline]
                        pub fn children(
                            &self,
                        ) -> ::planus::Result<
                            ::core::option::Option<
                                ::planus::Vector<'a, ::planus::Result<self::FieldRef<'a>>>,
                            >,
                        > {
                            self.0.access(5, "Field", "children")
                        }

                        /// Getter for the [`custom_metadata` field](Field#structfield.custom_metadata).
                        #[inline]
                        pub fn custom_metadata(
                            &self,
                        ) -> ::planus::Result<
                            ::core::option::Option<
                                ::planus::Vector<'a, ::planus::Result<self::KeyValueRef<'a>>>,
                            >,
                        > {
                            self.0.access(6, "Field", "custom_metadata")
                        }
                    }

                    impl<'a> ::core::fmt::Debug for FieldRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("FieldRef");
                            if let ::core::option::Option::Some(field_name) =
                                self.name().transpose()
                            {
                                f.field("name", &field_name);
                            }
                            f.field("nullable", &self.nullable());
                            if let ::core::option::Option::Some(field_type_) =
                                self.type_().transpose()
                            {
                                f.field("type_", &field_type_);
                            }
                            if let ::core::option::Option::Some(field_dictionary) =
                                self.dictionary().transpose()
                            {
                                f.field("dictionary", &field_dictionary);
                            }
                            if let ::core::option::Option::Some(field_children) =
                                self.children().transpose()
                            {
                                f.field("children", &field_children);
                            }
                            if let ::core::option::Option::Some(field_custom_metadata) =
                                self.custom_metadata().transpose()
                            {
                                f.field("custom_metadata", &field_custom_metadata);
                            }
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<FieldRef<'a>> for Field {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: FieldRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                name: value.name()?.map(::core::convert::Into::into),
                                nullable: ::core::convert::TryInto::try_into(value.nullable()?)?,
                                type_: if let ::core::option::Option::Some(type_) = value.type_()? {
                                    ::core::option::Option::Some(
                                        ::core::convert::TryInto::try_into(type_)?,
                                    )
                                } else {
                                    ::core::option::Option::None
                                },
                                dictionary: if let ::core::option::Option::Some(dictionary) =
                                    value.dictionary()?
                                {
                                    ::core::option::Option::Some(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryInto::try_into(dictionary)?,
                                    ))
                                } else {
                                    ::core::option::Option::None
                                },
                                children: if let ::core::option::Option::Some(children) =
                                    value.children()?
                                {
                                    ::core::option::Option::Some(children.to_vec_result()?)
                                } else {
                                    ::core::option::Option::None
                                },
                                custom_metadata: if let ::core::option::Option::Some(
                                    custom_metadata,
                                ) = value.custom_metadata()?
                                {
                                    ::core::option::Option::Some(custom_metadata.to_vec_result()?)
                                } else {
                                    ::core::option::Option::None
                                },
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for FieldRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for FieldRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[FieldRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<Field>> for Field {
                        type Value = ::planus::Offset<Field>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Field>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for FieldRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[FieldRef]", "read_as_root", 0)
                            })
                        }
                    }

                    ///  ----------------------------------------------------------------------
                    ///  Endianness of the platform producing the data
                    ///
                    /// Generated from these locations:
                    /// * Enum `Endianness` in the file `Schema.fbs:536`
                    #[derive(
                        Copy,
                        Clone,
                        Debug,
                        PartialEq,
                        Eq,
                        PartialOrd,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    #[repr(i16)]
                    pub enum Endianness {
                        /// The variant `Little` in the enum `Endianness`
                        Little = 0,

                        /// The variant `Big` in the enum `Endianness`
                        Big = 1,
                    }

                    impl Endianness {
                        /// Array containing all valid variants of Endianness
                        pub const ENUM_VALUES: [Self; 2] = [Self::Little, Self::Big];
                    }

                    impl ::core::convert::TryFrom<i16> for Endianness {
                        type Error = ::planus::errors::UnknownEnumTagKind;
                        #[inline]
                        fn try_from(
                            value: i16,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTagKind>
                        {
                            #[allow(clippy::match_single_binding)]
                            match value {
                                0 => ::core::result::Result::Ok(Endianness::Little),
                                1 => ::core::result::Result::Ok(Endianness::Big),

                                _ => ::core::result::Result::Err(
                                    ::planus::errors::UnknownEnumTagKind { tag: value as i128 },
                                ),
                            }
                        }
                    }

                    impl ::core::convert::From<Endianness> for i16 {
                        #[inline]
                        fn from(value: Endianness) -> Self {
                            value as i16
                        }
                    }

                    /// # Safety
                    /// The Planus compiler correctly calculates `ALIGNMENT` and `SIZE`.
                    unsafe impl ::planus::Primitive for Endianness {
                        const ALIGNMENT: usize = 2;
                        const SIZE: usize = 2;
                    }

                    impl ::planus::WriteAsPrimitive<Endianness> for Endianness {
                        #[inline]
                        fn write<const N: usize>(
                            &self,
                            cursor: ::planus::Cursor<'_, N>,
                            buffer_position: u32,
                        ) {
                            (*self as i16).write(cursor, buffer_position);
                        }
                    }

                    impl ::planus::WriteAs<Endianness> for Endianness {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Endianness {
                            *self
                        }
                    }

                    impl ::planus::WriteAsDefault<Endianness, Endianness> for Endianness {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                            default: &Endianness,
                        ) -> ::core::option::Option<Endianness> {
                            if self == default {
                                ::core::option::Option::None
                            } else {
                                ::core::option::Option::Some(*self)
                            }
                        }
                    }

                    impl ::planus::WriteAsOptional<Endianness> for Endianness {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<Endianness> {
                            ::core::option::Option::Some(*self)
                        }
                    }

                    impl<'buf> ::planus::TableRead<'buf> for Endianness {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            let n: i16 = ::planus::TableRead::from_buffer(buffer, offset)?;
                            ::core::result::Result::Ok(::core::convert::TryInto::try_into(n)?)
                        }
                    }

                    impl<'buf> ::planus::VectorReadInner<'buf> for Endianness {
                        type Error = ::planus::errors::UnknownEnumTag;
                        const STRIDE: usize = 2;
                        #[inline]
                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTag>
                        {
                            let value = unsafe {
                                <i16 as ::planus::VectorRead>::from_buffer(buffer, offset)
                            };
                            let value: ::core::result::Result<Self, _> =
                                ::core::convert::TryInto::try_into(value);
                            value.map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "Endianness",
                                    "VectorRead::from_buffer",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<Endianness> for Endianness {
                        const STRIDE: usize = 2;

                        type Value = Self;

                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self {
                            *self
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[Self],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 2];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (2 * i) as u32,
                                );
                            }
                        }
                    }

                    ///  ----------------------------------------------------------------------
                    ///  A Buffer represents a single contiguous memory segment
                    ///
                    /// Generated from these locations:
                    /// * Struct `Buffer` in the file `Schema.fbs:540`
                    #[derive(
                        Copy,
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        Default,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct Buffer {
                        ///  The relative offset into the shared memory page where the bytes for this
                        ///  buffer starts
                        pub offset: i64,

                        ///  The absolute length (in bytes) of the memory buffer. The memory is found
                        ///  from offset (inclusive) to offset + length (non-inclusive). When building
                        ///  messages using the encapsulated IPC message, padding bytes may be written
                        ///  after a buffer, but such padding bytes do not need to be accounted for in
                        ///  the size here.
                        pub length: i64,
                    }

                    /// # Safety
                    /// The Planus compiler correctly calculates `ALIGNMENT` and `SIZE`.
                    unsafe impl ::planus::Primitive for Buffer {
                        const ALIGNMENT: usize = 8;
                        const SIZE: usize = 16;
                    }

                    #[allow(clippy::identity_op)]
                    impl ::planus::WriteAsPrimitive<Buffer> for Buffer {
                        #[inline]
                        fn write<const N: usize>(
                            &self,
                            cursor: ::planus::Cursor<'_, N>,
                            buffer_position: u32,
                        ) {
                            let (cur, cursor) = cursor.split::<8, 8>();
                            self.offset.write(cur, buffer_position - 0);
                            let (cur, cursor) = cursor.split::<8, 0>();
                            self.length.write(cur, buffer_position - 8);
                            cursor.finish([]);
                        }
                    }

                    impl ::planus::WriteAsOffset<Buffer> for Buffer {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Buffer> {
                            unsafe {
                                builder.write_with(16, 7, |buffer_position, bytes| {
                                    let bytes = bytes.as_mut_ptr();

                                    ::planus::WriteAsPrimitive::write(
                                        self,
                                        ::planus::Cursor::new(
                                            &mut *(bytes
                                                as *mut [::core::mem::MaybeUninit<u8>; 16]),
                                        ),
                                        buffer_position,
                                    );
                                });
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<Buffer> for Buffer {
                        type Prepared = Self;
                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self {
                            *self
                        }
                    }

                    impl ::planus::WriteAsOptional<Buffer> for Buffer {
                        type Prepared = Self;
                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<Self> {
                            ::core::option::Option::Some(*self)
                        }
                    }

                    /// Reference to a deserialized [Buffer].
                    #[derive(Copy, Clone)]
                    pub struct BufferRef<'a>(::planus::ArrayWithStartOffset<'a, 16>);

                    impl<'a> BufferRef<'a> {
                        /// Getter for the [`offset` field](Buffer#structfield.offset).
                        pub fn offset(&self) -> i64 {
                            let buffer = self.0.advance_as_array::<8>(0).unwrap();

                            i64::from_le_bytes(*buffer.as_array())
                        }

                        /// Getter for the [`length` field](Buffer#structfield.length).
                        pub fn length(&self) -> i64 {
                            let buffer = self.0.advance_as_array::<8>(8).unwrap();

                            i64::from_le_bytes(*buffer.as_array())
                        }
                    }

                    impl<'a> ::core::fmt::Debug for BufferRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("BufferRef");
                            f.field("offset", &self.offset());
                            f.field("length", &self.length());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::From<::planus::ArrayWithStartOffset<'a, 16>> for BufferRef<'a> {
                        fn from(array: ::planus::ArrayWithStartOffset<'a, 16>) -> Self {
                            Self(array)
                        }
                    }

                    impl<'a> ::core::convert::From<BufferRef<'a>> for Buffer {
                        #[allow(unreachable_code)]
                        fn from(value: BufferRef<'a>) -> Self {
                            Self {
                                offset: value.offset(),
                                length: value.length(),
                            }
                        }
                    }

                    impl<'a, 'b> ::core::cmp::PartialEq<BufferRef<'a>> for BufferRef<'b> {
                        fn eq(&self, other: &BufferRef<'_>) -> bool {
                            self.offset() == other.offset() && self.length() == other.length()
                        }
                    }

                    impl<'a> ::core::cmp::Eq for BufferRef<'a> {}
                    impl<'a, 'b> ::core::cmp::PartialOrd<BufferRef<'a>> for BufferRef<'b> {
                        fn partial_cmp(
                            &self,
                            other: &BufferRef<'_>,
                        ) -> ::core::option::Option<::core::cmp::Ordering> {
                            ::core::option::Option::Some(::core::cmp::Ord::cmp(self, other))
                        }
                    }

                    impl<'a> ::core::cmp::Ord for BufferRef<'a> {
                        fn cmp(&self, other: &BufferRef<'_>) -> ::core::cmp::Ordering {
                            self.offset()
                                .cmp(&other.offset())
                                .then_with(|| self.length().cmp(&other.length()))
                        }
                    }

                    impl<'a> ::core::hash::Hash for BufferRef<'a> {
                        fn hash<H: ::core::hash::Hasher>(&self, state: &mut H) {
                            self.offset().hash(state);
                            self.length().hash(state);
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for BufferRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            let buffer = buffer.advance_as_array::<16>(offset)?;
                            ::core::result::Result::Ok(Self(buffer))
                        }
                    }

                    impl<'a> ::planus::VectorRead<'a> for BufferRef<'a> {
                        const STRIDE: usize = 16;

                        #[inline]
                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> Self {
                            Self(unsafe { buffer.unchecked_advance_as_array(offset) })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<Buffer> for Buffer {
                        const STRIDE: usize = 16;

                        type Value = Buffer;

                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self::Value {
                            *self
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[Buffer],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 16];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (16 * i) as u32,
                                );
                            }
                        }
                    }

                    ///  ----------------------------------------------------------------------
                    ///  A Schema describes the columns in a row batch
                    ///
                    /// Generated from these locations:
                    /// * Table `Schema` in the file `Schema.fbs:556`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct Schema {
                        ///  endianness of the buffer
                        ///  it is Little Endian by default
                        ///  if endianness doesn't match the underlying system then the vectors need to be converted
                        pub endianness: self::Endianness,
                        /// The field `fields` in the table `Schema`
                        pub fields: ::core::option::Option<::planus::alloc::vec::Vec<self::Field>>,
                        /// The field `custom_metadata` in the table `Schema`
                        pub custom_metadata:
                            ::core::option::Option<::planus::alloc::vec::Vec<self::KeyValue>>,
                        ///  Features used in the stream/file.
                        pub features:
                            ::core::option::Option<::planus::alloc::vec::Vec<self::Feature>>,
                    }

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for Schema {
                        fn default() -> Self {
                            Self {
                                endianness: self::Endianness::Little,
                                fields: ::core::default::Default::default(),
                                custom_metadata: ::core::default::Default::default(),
                                features: ::core::default::Default::default(),
                            }
                        }
                    }

                    impl Schema {
                        /// Creates a [SchemaBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> SchemaBuilder<()> {
                            SchemaBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            field_endianness: impl ::planus::WriteAsDefault<
                                self::Endianness,
                                self::Endianness,
                            >,
                            field_fields: impl ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::Field>]>,
                            >,
                            field_custom_metadata: impl ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::KeyValue>]>,
                            >,
                            field_features: impl ::planus::WriteAsOptional<
                                ::planus::Offset<[self::Feature]>,
                            >,
                        ) -> ::planus::Offset<Self> {
                            let prepared_endianness =
                                field_endianness.prepare(builder, &self::Endianness::Little);
                            let prepared_fields = field_fields.prepare(builder);
                            let prepared_custom_metadata = field_custom_metadata.prepare(builder);
                            let prepared_features = field_features.prepare(builder);

                            let mut table_writer: ::planus::table_writer::TableWriter<12> =
                                ::core::default::Default::default();
                            if prepared_fields.is_some() {
                                table_writer.write_entry::<::planus::Offset<[::planus::Offset<self::Field>]>>(1);
                            }
                            if prepared_custom_metadata.is_some() {
                                table_writer.write_entry::<::planus::Offset<[::planus::Offset<self::KeyValue>]>>(2);
                            }
                            if prepared_features.is_some() {
                                table_writer.write_entry::<::planus::Offset<[self::Feature]>>(3);
                            }
                            if prepared_endianness.is_some() {
                                table_writer.write_entry::<self::Endianness>(0);
                            }

                            unsafe {
                                table_writer.finish(builder, |object_writer| {
                                    if let ::core::option::Option::Some(prepared_fields) =
                                        prepared_fields
                                    {
                                        object_writer.write::<_, _, 4>(&prepared_fields);
                                    }
                                    if let ::core::option::Option::Some(prepared_custom_metadata) =
                                        prepared_custom_metadata
                                    {
                                        object_writer.write::<_, _, 4>(&prepared_custom_metadata);
                                    }
                                    if let ::core::option::Option::Some(prepared_features) =
                                        prepared_features
                                    {
                                        object_writer.write::<_, _, 4>(&prepared_features);
                                    }
                                    if let ::core::option::Option::Some(prepared_endianness) =
                                        prepared_endianness
                                    {
                                        object_writer.write::<_, _, 2>(&prepared_endianness);
                                    }
                                });
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Schema>> for Schema {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Schema> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Schema>> for Schema {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Schema>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Schema> for Schema {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Schema> {
                            Schema::create(
                                builder,
                                self.endianness,
                                &self.fields,
                                &self.custom_metadata,
                                &self.features,
                            )
                        }
                    }

                    /// Builder for serializing an instance of the [Schema] type.
                    ///
                    /// Can be created using the [Schema::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct SchemaBuilder<State>(State);

                    impl SchemaBuilder<()> {
                        /// Setter for the [`endianness` field](Schema#structfield.endianness).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn endianness<T0>(self, value: T0) -> SchemaBuilder<(T0,)>
                        where
                            T0: ::planus::WriteAsDefault<self::Endianness, self::Endianness>,
                        {
                            SchemaBuilder((value,))
                        }

                        /// Sets the [`endianness` field](Schema#structfield.endianness) to the default value.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn endianness_as_default(
                            self,
                        ) -> SchemaBuilder<(::planus::DefaultValue,)> {
                            self.endianness(::planus::DefaultValue)
                        }
                    }

                    impl<T0> SchemaBuilder<(T0,)> {
                        /// Setter for the [`fields` field](Schema#structfield.fields).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn fields<T1>(self, value: T1) -> SchemaBuilder<(T0, T1)>
                        where
                            T1: ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::Field>]>,
                            >,
                        {
                            let (v0,) = self.0;
                            SchemaBuilder((v0, value))
                        }

                        /// Sets the [`fields` field](Schema#structfield.fields) to null.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn fields_as_null(self) -> SchemaBuilder<(T0, ())> {
                            self.fields(())
                        }
                    }

                    impl<T0, T1> SchemaBuilder<(T0, T1)> {
                        /// Setter for the [`custom_metadata` field](Schema#structfield.custom_metadata).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn custom_metadata<T2>(self, value: T2) -> SchemaBuilder<(T0, T1, T2)>
                        where
                            T2: ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::KeyValue>]>,
                            >,
                        {
                            let (v0, v1) = self.0;
                            SchemaBuilder((v0, v1, value))
                        }

                        /// Sets the [`custom_metadata` field](Schema#structfield.custom_metadata) to null.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn custom_metadata_as_null(self) -> SchemaBuilder<(T0, T1, ())> {
                            self.custom_metadata(())
                        }
                    }

                    impl<T0, T1, T2> SchemaBuilder<(T0, T1, T2)> {
                        /// Setter for the [`features` field](Schema#structfield.features).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn features<T3>(self, value: T3) -> SchemaBuilder<(T0, T1, T2, T3)>
                        where
                            T3: ::planus::WriteAsOptional<::planus::Offset<[self::Feature]>>,
                        {
                            let (v0, v1, v2) = self.0;
                            SchemaBuilder((v0, v1, v2, value))
                        }

                        /// Sets the [`features` field](Schema#structfield.features) to null.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn features_as_null(self) -> SchemaBuilder<(T0, T1, T2, ())> {
                            self.features(())
                        }
                    }

                    impl<T0, T1, T2, T3> SchemaBuilder<(T0, T1, T2, T3)> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Schema].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Schema>
                        where
                            Self: ::planus::WriteAsOffset<Schema>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<self::Endianness, self::Endianness>,
                            T1: ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::Field>]>,
                            >,
                            T2: ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::KeyValue>]>,
                            >,
                            T3: ::planus::WriteAsOptional<::planus::Offset<[self::Feature]>>,
                        > ::planus::WriteAs<::planus::Offset<Schema>>
                        for SchemaBuilder<(T0, T1, T2, T3)>
                    {
                        type Prepared = ::planus::Offset<Schema>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Schema> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<self::Endianness, self::Endianness>,
                            T1: ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::Field>]>,
                            >,
                            T2: ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::KeyValue>]>,
                            >,
                            T3: ::planus::WriteAsOptional<::planus::Offset<[self::Feature]>>,
                        > ::planus::WriteAsOptional<::planus::Offset<Schema>>
                        for SchemaBuilder<(T0, T1, T2, T3)>
                    {
                        type Prepared = ::planus::Offset<Schema>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Schema>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<self::Endianness, self::Endianness>,
                            T1: ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::Field>]>,
                            >,
                            T2: ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::KeyValue>]>,
                            >,
                            T3: ::planus::WriteAsOptional<::planus::Offset<[self::Feature]>>,
                        > ::planus::WriteAsOffset<Schema> for SchemaBuilder<(T0, T1, T2, T3)>
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Schema> {
                            let (v0, v1, v2, v3) = &self.0;
                            Schema::create(builder, v0, v1, v2, v3)
                        }
                    }

                    /// Reference to a deserialized [Schema].
                    #[derive(Copy, Clone)]
                    pub struct SchemaRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> SchemaRef<'a> {
                        /// Getter for the [`endianness` field](Schema#structfield.endianness).
                        #[inline]
                        pub fn endianness(&self) -> ::planus::Result<self::Endianness> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(0, "Schema", "endianness")?
                                    .unwrap_or(self::Endianness::Little),
                            )
                        }

                        /// Getter for the [`fields` field](Schema#structfield.fields).
                        #[inline]
                        pub fn fields(
                            &self,
                        ) -> ::planus::Result<
                            ::core::option::Option<
                                ::planus::Vector<'a, ::planus::Result<self::FieldRef<'a>>>,
                            >,
                        > {
                            self.0.access(1, "Schema", "fields")
                        }

                        /// Getter for the [`custom_metadata` field](Schema#structfield.custom_metadata).
                        #[inline]
                        pub fn custom_metadata(
                            &self,
                        ) -> ::planus::Result<
                            ::core::option::Option<
                                ::planus::Vector<'a, ::planus::Result<self::KeyValueRef<'a>>>,
                            >,
                        > {
                            self.0.access(2, "Schema", "custom_metadata")
                        }

                        /// Getter for the [`features` field](Schema#structfield.features).
                        #[inline]
                        pub fn features(
                            &self,
                        ) -> ::planus::Result<
                            ::core::option::Option<
                                ::planus::Vector<
                                    'a,
                                    ::core::result::Result<
                                        self::Feature,
                                        ::planus::errors::UnknownEnumTag,
                                    >,
                                >,
                            >,
                        > {
                            self.0.access(3, "Schema", "features")
                        }
                    }

                    impl<'a> ::core::fmt::Debug for SchemaRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("SchemaRef");
                            f.field("endianness", &self.endianness());
                            if let ::core::option::Option::Some(field_fields) =
                                self.fields().transpose()
                            {
                                f.field("fields", &field_fields);
                            }
                            if let ::core::option::Option::Some(field_custom_metadata) =
                                self.custom_metadata().transpose()
                            {
                                f.field("custom_metadata", &field_custom_metadata);
                            }
                            if let ::core::option::Option::Some(field_features) =
                                self.features().transpose()
                            {
                                f.field("features", &field_features);
                            }
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<SchemaRef<'a>> for Schema {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: SchemaRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                endianness: ::core::convert::TryInto::try_into(
                                    value.endianness()?,
                                )?,
                                fields: if let ::core::option::Option::Some(fields) =
                                    value.fields()?
                                {
                                    ::core::option::Option::Some(fields.to_vec_result()?)
                                } else {
                                    ::core::option::Option::None
                                },
                                custom_metadata: if let ::core::option::Option::Some(
                                    custom_metadata,
                                ) = value.custom_metadata()?
                                {
                                    ::core::option::Option::Some(custom_metadata.to_vec_result()?)
                                } else {
                                    ::core::option::Option::None
                                },
                                features: if let ::core::option::Option::Some(features) =
                                    value.features()?
                                {
                                    ::core::option::Option::Some(features.to_vec_result()?)
                                } else {
                                    ::core::option::Option::None
                                },
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for SchemaRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for SchemaRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[SchemaRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<Schema>> for Schema {
                        type Value = ::planus::Offset<Schema>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Schema>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for SchemaRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[SchemaRef]", "read_as_root", 0)
                            })
                        }
                    }

                    ///  ----------------------------------------------------------------------
                    ///  Data structures for describing a table row batch (a collection of
                    ///  equal-length Arrow arrays)
                    ///  Metadata about a field at some level of a nested type tree (but not
                    ///  its children).
                    ///
                    ///  For example, a List<Int16> with values `[[1, 2, 3], null, [4], [5, 6], null]`
                    ///  would have {length: 5, null_count: 2} for its List node, and {length: 6,
                    ///  null_count: 0} for its Int16 node, as separate FieldNode structs
                    ///
                    /// Generated from these locations:
                    /// * Struct `FieldNode` in the file `Message.fbs:34`
                    #[derive(
                        Copy,
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        Default,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct FieldNode {
                        ///  The number of value slots in the Arrow array at this level of a nested
                        ///  tree
                        pub length: i64,

                        ///  The number of observed nulls. Fields with null_count == 0 may choose not
                        ///  to write their physical validity bitmap out as a materialized buffer,
                        ///  instead setting the length of the bitmap buffer to 0.
                        pub null_count: i64,
                    }

                    /// # Safety
                    /// The Planus compiler correctly calculates `ALIGNMENT` and `SIZE`.
                    unsafe impl ::planus::Primitive for FieldNode {
                        const ALIGNMENT: usize = 8;
                        const SIZE: usize = 16;
                    }

                    #[allow(clippy::identity_op)]
                    impl ::planus::WriteAsPrimitive<FieldNode> for FieldNode {
                        #[inline]
                        fn write<const N: usize>(
                            &self,
                            cursor: ::planus::Cursor<'_, N>,
                            buffer_position: u32,
                        ) {
                            let (cur, cursor) = cursor.split::<8, 8>();
                            self.length.write(cur, buffer_position - 0);
                            let (cur, cursor) = cursor.split::<8, 0>();
                            self.null_count.write(cur, buffer_position - 8);
                            cursor.finish([]);
                        }
                    }

                    impl ::planus::WriteAsOffset<FieldNode> for FieldNode {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<FieldNode> {
                            unsafe {
                                builder.write_with(16, 7, |buffer_position, bytes| {
                                    let bytes = bytes.as_mut_ptr();

                                    ::planus::WriteAsPrimitive::write(
                                        self,
                                        ::planus::Cursor::new(
                                            &mut *(bytes
                                                as *mut [::core::mem::MaybeUninit<u8>; 16]),
                                        ),
                                        buffer_position,
                                    );
                                });
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<FieldNode> for FieldNode {
                        type Prepared = Self;
                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self {
                            *self
                        }
                    }

                    impl ::planus::WriteAsOptional<FieldNode> for FieldNode {
                        type Prepared = Self;
                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<Self> {
                            ::core::option::Option::Some(*self)
                        }
                    }

                    /// Reference to a deserialized [FieldNode].
                    #[derive(Copy, Clone)]
                    pub struct FieldNodeRef<'a>(::planus::ArrayWithStartOffset<'a, 16>);

                    impl<'a> FieldNodeRef<'a> {
                        /// Getter for the [`length` field](FieldNode#structfield.length).
                        pub fn length(&self) -> i64 {
                            let buffer = self.0.advance_as_array::<8>(0).unwrap();

                            i64::from_le_bytes(*buffer.as_array())
                        }

                        /// Getter for the [`null_count` field](FieldNode#structfield.null_count).
                        pub fn null_count(&self) -> i64 {
                            let buffer = self.0.advance_as_array::<8>(8).unwrap();

                            i64::from_le_bytes(*buffer.as_array())
                        }
                    }

                    impl<'a> ::core::fmt::Debug for FieldNodeRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("FieldNodeRef");
                            f.field("length", &self.length());
                            f.field("null_count", &self.null_count());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::From<::planus::ArrayWithStartOffset<'a, 16>> for FieldNodeRef<'a> {
                        fn from(array: ::planus::ArrayWithStartOffset<'a, 16>) -> Self {
                            Self(array)
                        }
                    }

                    impl<'a> ::core::convert::From<FieldNodeRef<'a>> for FieldNode {
                        #[allow(unreachable_code)]
                        fn from(value: FieldNodeRef<'a>) -> Self {
                            Self {
                                length: value.length(),
                                null_count: value.null_count(),
                            }
                        }
                    }

                    impl<'a, 'b> ::core::cmp::PartialEq<FieldNodeRef<'a>> for FieldNodeRef<'b> {
                        fn eq(&self, other: &FieldNodeRef<'_>) -> bool {
                            self.length() == other.length()
                                && self.null_count() == other.null_count()
                        }
                    }

                    impl<'a> ::core::cmp::Eq for FieldNodeRef<'a> {}
                    impl<'a, 'b> ::core::cmp::PartialOrd<FieldNodeRef<'a>> for FieldNodeRef<'b> {
                        fn partial_cmp(
                            &self,
                            other: &FieldNodeRef<'_>,
                        ) -> ::core::option::Option<::core::cmp::Ordering> {
                            ::core::option::Option::Some(::core::cmp::Ord::cmp(self, other))
                        }
                    }

                    impl<'a> ::core::cmp::Ord for FieldNodeRef<'a> {
                        fn cmp(&self, other: &FieldNodeRef<'_>) -> ::core::cmp::Ordering {
                            self.length()
                                .cmp(&other.length())
                                .then_with(|| self.null_count().cmp(&other.null_count()))
                        }
                    }

                    impl<'a> ::core::hash::Hash for FieldNodeRef<'a> {
                        fn hash<H: ::core::hash::Hasher>(&self, state: &mut H) {
                            self.length().hash(state);
                            self.null_count().hash(state);
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for FieldNodeRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            let buffer = buffer.advance_as_array::<16>(offset)?;
                            ::core::result::Result::Ok(Self(buffer))
                        }
                    }

                    impl<'a> ::planus::VectorRead<'a> for FieldNodeRef<'a> {
                        const STRIDE: usize = 16;

                        #[inline]
                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> Self {
                            Self(unsafe { buffer.unchecked_advance_as_array(offset) })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<FieldNode> for FieldNode {
                        const STRIDE: usize = 16;

                        type Value = FieldNode;

                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self::Value {
                            *self
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[FieldNode],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 16];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (16 * i) as u32,
                                );
                            }
                        }
                    }

                    /// The enum `CompressionType` in the namespace `org.apache.arrow.flatbuf`
                    ///
                    /// Generated from these locations:
                    /// * Enum `CompressionType` in the file `Message.fbs:45`
                    #[derive(
                        Copy,
                        Clone,
                        Debug,
                        PartialEq,
                        Eq,
                        PartialOrd,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    #[repr(i8)]
                    pub enum CompressionType {
                        /// The variant `LZ4_FRAME` in the enum `CompressionType`
                        Lz4Frame = 0,

                        /// The variant `ZSTD` in the enum `CompressionType`
                        Zstd = 1,
                    }

                    impl CompressionType {
                        /// Array containing all valid variants of CompressionType
                        pub const ENUM_VALUES: [Self; 2] = [Self::Lz4Frame, Self::Zstd];
                    }

                    impl ::core::convert::TryFrom<i8> for CompressionType {
                        type Error = ::planus::errors::UnknownEnumTagKind;
                        #[inline]
                        fn try_from(
                            value: i8,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTagKind>
                        {
                            #[allow(clippy::match_single_binding)]
                            match value {
                                0 => ::core::result::Result::Ok(CompressionType::Lz4Frame),
                                1 => ::core::result::Result::Ok(CompressionType::Zstd),

                                _ => ::core::result::Result::Err(
                                    ::planus::errors::UnknownEnumTagKind { tag: value as i128 },
                                ),
                            }
                        }
                    }

                    impl ::core::convert::From<CompressionType> for i8 {
                        #[inline]
                        fn from(value: CompressionType) -> Self {
                            value as i8
                        }
                    }

                    /// # Safety
                    /// The Planus compiler correctly calculates `ALIGNMENT` and `SIZE`.
                    unsafe impl ::planus::Primitive for CompressionType {
                        const ALIGNMENT: usize = 1;
                        const SIZE: usize = 1;
                    }

                    impl ::planus::WriteAsPrimitive<CompressionType> for CompressionType {
                        #[inline]
                        fn write<const N: usize>(
                            &self,
                            cursor: ::planus::Cursor<'_, N>,
                            buffer_position: u32,
                        ) {
                            (*self as i8).write(cursor, buffer_position);
                        }
                    }

                    impl ::planus::WriteAs<CompressionType> for CompressionType {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> CompressionType {
                            *self
                        }
                    }

                    impl ::planus::WriteAsDefault<CompressionType, CompressionType> for CompressionType {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                            default: &CompressionType,
                        ) -> ::core::option::Option<CompressionType> {
                            if self == default {
                                ::core::option::Option::None
                            } else {
                                ::core::option::Option::Some(*self)
                            }
                        }
                    }

                    impl ::planus::WriteAsOptional<CompressionType> for CompressionType {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<CompressionType> {
                            ::core::option::Option::Some(*self)
                        }
                    }

                    impl<'buf> ::planus::TableRead<'buf> for CompressionType {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            let n: i8 = ::planus::TableRead::from_buffer(buffer, offset)?;
                            ::core::result::Result::Ok(::core::convert::TryInto::try_into(n)?)
                        }
                    }

                    impl<'buf> ::planus::VectorReadInner<'buf> for CompressionType {
                        type Error = ::planus::errors::UnknownEnumTag;
                        const STRIDE: usize = 1;
                        #[inline]
                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTag>
                        {
                            let value = unsafe { *buffer.buffer.get_unchecked(offset) as i8 };
                            let value: ::core::result::Result<Self, _> =
                                ::core::convert::TryInto::try_into(value);
                            value.map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "CompressionType",
                                    "VectorRead::from_buffer",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<CompressionType> for CompressionType {
                        const STRIDE: usize = 1;

                        type Value = Self;

                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self {
                            *self
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[Self],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 1];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - i as u32,
                                );
                            }
                        }
                    }

                    ///  Provided for forward compatibility in case we need to support different
                    ///  strategies for compressing the IPC message body (like whole-body
                    ///  compression rather than buffer-level) in the future
                    ///
                    /// Generated from these locations:
                    /// * Enum `BodyCompressionMethod` in the file `Message.fbs:58`
                    #[derive(
                        Copy,
                        Clone,
                        Debug,
                        PartialEq,
                        Eq,
                        PartialOrd,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    #[repr(i8)]
                    pub enum BodyCompressionMethod {
                        ///  Each constituent buffer is first compressed with the indicated
                        ///  compressor, and then written with the uncompressed length in the first 8
                        ///  bytes as a 64-bit little-endian signed integer followed by the compressed
                        ///  buffer bytes (and then padding as required by the protocol). The
                        ///  uncompressed length may be set to -1 to indicate that the data that
                        ///  follows is not compressed, which can be useful for cases where
                        ///  compression does not yield appreciable savings.
                        Buffer = 0,
                    }

                    impl BodyCompressionMethod {
                        /// Array containing all valid variants of BodyCompressionMethod
                        pub const ENUM_VALUES: [Self; 1] = [Self::Buffer];
                    }

                    impl ::core::convert::TryFrom<i8> for BodyCompressionMethod {
                        type Error = ::planus::errors::UnknownEnumTagKind;
                        #[inline]
                        fn try_from(
                            value: i8,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTagKind>
                        {
                            #[allow(clippy::match_single_binding)]
                            match value {
                                0 => ::core::result::Result::Ok(BodyCompressionMethod::Buffer),

                                _ => ::core::result::Result::Err(
                                    ::planus::errors::UnknownEnumTagKind { tag: value as i128 },
                                ),
                            }
                        }
                    }

                    impl ::core::convert::From<BodyCompressionMethod> for i8 {
                        #[inline]
                        fn from(value: BodyCompressionMethod) -> Self {
                            value as i8
                        }
                    }

                    /// # Safety
                    /// The Planus compiler correctly calculates `ALIGNMENT` and `SIZE`.
                    unsafe impl ::planus::Primitive for BodyCompressionMethod {
                        const ALIGNMENT: usize = 1;
                        const SIZE: usize = 1;
                    }

                    impl ::planus::WriteAsPrimitive<BodyCompressionMethod> for BodyCompressionMethod {
                        #[inline]
                        fn write<const N: usize>(
                            &self,
                            cursor: ::planus::Cursor<'_, N>,
                            buffer_position: u32,
                        ) {
                            (*self as i8).write(cursor, buffer_position);
                        }
                    }

                    impl ::planus::WriteAs<BodyCompressionMethod> for BodyCompressionMethod {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                        ) -> BodyCompressionMethod {
                            *self
                        }
                    }

                    impl ::planus::WriteAsDefault<BodyCompressionMethod, BodyCompressionMethod>
                        for BodyCompressionMethod
                    {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                            default: &BodyCompressionMethod,
                        ) -> ::core::option::Option<BodyCompressionMethod> {
                            if self == default {
                                ::core::option::Option::None
                            } else {
                                ::core::option::Option::Some(*self)
                            }
                        }
                    }

                    impl ::planus::WriteAsOptional<BodyCompressionMethod> for BodyCompressionMethod {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<BodyCompressionMethod> {
                            ::core::option::Option::Some(*self)
                        }
                    }

                    impl<'buf> ::planus::TableRead<'buf> for BodyCompressionMethod {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            let n: i8 = ::planus::TableRead::from_buffer(buffer, offset)?;
                            ::core::result::Result::Ok(::core::convert::TryInto::try_into(n)?)
                        }
                    }

                    impl<'buf> ::planus::VectorReadInner<'buf> for BodyCompressionMethod {
                        type Error = ::planus::errors::UnknownEnumTag;
                        const STRIDE: usize = 1;
                        #[inline]
                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTag>
                        {
                            let value = unsafe { *buffer.buffer.get_unchecked(offset) as i8 };
                            let value: ::core::result::Result<Self, _> =
                                ::core::convert::TryInto::try_into(value);
                            value.map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "BodyCompressionMethod",
                                    "VectorRead::from_buffer",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<BodyCompressionMethod> for BodyCompressionMethod {
                        const STRIDE: usize = 1;

                        type Value = Self;

                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self {
                            *self
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[Self],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 1];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - i as u32,
                                );
                            }
                        }
                    }

                    ///  Optional compression for the memory buffers constituting IPC message
                    ///  bodies. Intended for use with RecordBatch but could be used for other
                    ///  message types
                    ///
                    /// Generated from these locations:
                    /// * Table `BodyCompression` in the file `Message.fbs:72`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct BodyCompression {
                        ///  Compressor library.
                        ///  For LZ4_FRAME, each compressed buffer must consist of a single frame.
                        pub codec: self::CompressionType,
                        ///  Indicates the way the record batch body was compressed
                        pub method: self::BodyCompressionMethod,
                    }

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for BodyCompression {
                        fn default() -> Self {
                            Self {
                                codec: self::CompressionType::Lz4Frame,
                                method: self::BodyCompressionMethod::Buffer,
                            }
                        }
                    }

                    impl BodyCompression {
                        /// Creates a [BodyCompressionBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> BodyCompressionBuilder<()> {
                            BodyCompressionBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            field_codec: impl ::planus::WriteAsDefault<
                                self::CompressionType,
                                self::CompressionType,
                            >,
                            field_method: impl ::planus::WriteAsDefault<
                                self::BodyCompressionMethod,
                                self::BodyCompressionMethod,
                            >,
                        ) -> ::planus::Offset<Self> {
                            let prepared_codec =
                                field_codec.prepare(builder, &self::CompressionType::Lz4Frame);
                            let prepared_method =
                                field_method.prepare(builder, &self::BodyCompressionMethod::Buffer);

                            let mut table_writer: ::planus::table_writer::TableWriter<8> =
                                ::core::default::Default::default();
                            if prepared_codec.is_some() {
                                table_writer.write_entry::<self::CompressionType>(0);
                            }
                            if prepared_method.is_some() {
                                table_writer.write_entry::<self::BodyCompressionMethod>(1);
                            }

                            unsafe {
                                table_writer.finish(builder, |object_writer| {
                                    if let ::core::option::Option::Some(prepared_codec) =
                                        prepared_codec
                                    {
                                        object_writer.write::<_, _, 1>(&prepared_codec);
                                    }
                                    if let ::core::option::Option::Some(prepared_method) =
                                        prepared_method
                                    {
                                        object_writer.write::<_, _, 1>(&prepared_method);
                                    }
                                });
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<BodyCompression>> for BodyCompression {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<BodyCompression> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<BodyCompression>> for BodyCompression {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<BodyCompression>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<BodyCompression> for BodyCompression {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<BodyCompression> {
                            BodyCompression::create(builder, self.codec, self.method)
                        }
                    }

                    /// Builder for serializing an instance of the [BodyCompression] type.
                    ///
                    /// Can be created using the [BodyCompression::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct BodyCompressionBuilder<State>(State);

                    impl BodyCompressionBuilder<()> {
                        /// Setter for the [`codec` field](BodyCompression#structfield.codec).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn codec<T0>(self, value: T0) -> BodyCompressionBuilder<(T0,)>
                        where
                            T0: ::planus::WriteAsDefault<
                                self::CompressionType,
                                self::CompressionType,
                            >,
                        {
                            BodyCompressionBuilder((value,))
                        }

                        /// Sets the [`codec` field](BodyCompression#structfield.codec) to the default value.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn codec_as_default(
                            self,
                        ) -> BodyCompressionBuilder<(::planus::DefaultValue,)>
                        {
                            self.codec(::planus::DefaultValue)
                        }
                    }

                    impl<T0> BodyCompressionBuilder<(T0,)> {
                        /// Setter for the [`method` field](BodyCompression#structfield.method).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn method<T1>(self, value: T1) -> BodyCompressionBuilder<(T0, T1)>
                        where
                            T1: ::planus::WriteAsDefault<
                                self::BodyCompressionMethod,
                                self::BodyCompressionMethod,
                            >,
                        {
                            let (v0,) = self.0;
                            BodyCompressionBuilder((v0, value))
                        }

                        /// Sets the [`method` field](BodyCompression#structfield.method) to the default value.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn method_as_default(
                            self,
                        ) -> BodyCompressionBuilder<(T0, ::planus::DefaultValue)>
                        {
                            self.method(::planus::DefaultValue)
                        }
                    }

                    impl<T0, T1> BodyCompressionBuilder<(T0, T1)> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [BodyCompression].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<BodyCompression>
                        where
                            Self: ::planus::WriteAsOffset<BodyCompression>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<self::CompressionType, self::CompressionType>,
                            T1: ::planus::WriteAsDefault<
                                self::BodyCompressionMethod,
                                self::BodyCompressionMethod,
                            >,
                        > ::planus::WriteAs<::planus::Offset<BodyCompression>>
                        for BodyCompressionBuilder<(T0, T1)>
                    {
                        type Prepared = ::planus::Offset<BodyCompression>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<BodyCompression> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<self::CompressionType, self::CompressionType>,
                            T1: ::planus::WriteAsDefault<
                                self::BodyCompressionMethod,
                                self::BodyCompressionMethod,
                            >,
                        >
                        ::planus::WriteAsOptional<::planus::Offset<BodyCompression>>
                        for BodyCompressionBuilder<(T0, T1)>
                    {
                        type Prepared = ::planus::Offset<BodyCompression>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<BodyCompression>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<self::CompressionType, self::CompressionType>,
                            T1: ::planus::WriteAsDefault<
                                self::BodyCompressionMethod,
                                self::BodyCompressionMethod,
                            >,
                        > ::planus::WriteAsOffset<BodyCompression>
                        for BodyCompressionBuilder<(T0, T1)>
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<BodyCompression> {
                            let (v0, v1) = &self.0;
                            BodyCompression::create(builder, v0, v1)
                        }
                    }

                    /// Reference to a deserialized [BodyCompression].
                    #[derive(Copy, Clone)]
                    pub struct BodyCompressionRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> BodyCompressionRef<'a> {
                        /// Getter for the [`codec` field](BodyCompression#structfield.codec).
                        #[inline]
                        pub fn codec(&self) -> ::planus::Result<self::CompressionType> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(0, "BodyCompression", "codec")?
                                    .unwrap_or(self::CompressionType::Lz4Frame),
                            )
                        }

                        /// Getter for the [`method` field](BodyCompression#structfield.method).
                        #[inline]
                        pub fn method(&self) -> ::planus::Result<self::BodyCompressionMethod> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(1, "BodyCompression", "method")?
                                    .unwrap_or(self::BodyCompressionMethod::Buffer),
                            )
                        }
                    }

                    impl<'a> ::core::fmt::Debug for BodyCompressionRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("BodyCompressionRef");
                            f.field("codec", &self.codec());
                            f.field("method", &self.method());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<BodyCompressionRef<'a>> for BodyCompression {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: BodyCompressionRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                codec: ::core::convert::TryInto::try_into(value.codec()?)?,
                                method: ::core::convert::TryInto::try_into(value.method()?)?,
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for BodyCompressionRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for BodyCompressionRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[BodyCompressionRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<BodyCompression>> for BodyCompression {
                        type Value = ::planus::Offset<BodyCompression>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<BodyCompression>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for BodyCompressionRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[BodyCompressionRef]",
                                    "read_as_root",
                                    0,
                                )
                            })
                        }
                    }

                    ///  A data header describing the shared memory layout of a "record" or "row"
                    ///  batch. Some systems call this a "row batch" internally and others a "record
                    ///  batch".
                    ///
                    /// Generated from these locations:
                    /// * Table `RecordBatch` in the file `Message.fbs:84`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct RecordBatch {
                        ///  number of records / rows. The arrays in the batch should all have this
                        ///  length
                        pub length: i64,
                        ///  Nodes correspond to the pre-ordered flattened logical schema
                        pub nodes:
                            ::core::option::Option<::planus::alloc::vec::Vec<self::FieldNode>>,
                        ///  Buffers correspond to the pre-ordered flattened buffer tree
                        ///
                        ///  The number of buffers appended to this list depends on the schema. For
                        ///  example, most primitive arrays will have 2 buffers, 1 for the validity
                        ///  bitmap and 1 for the values. For struct arrays, there will only be a
                        ///  single buffer for the validity (nulls) bitmap
                        pub buffers:
                            ::core::option::Option<::planus::alloc::vec::Vec<self::Buffer>>,
                        ///  Optional compression of the message body
                        pub compression: ::core::option::Option<
                            ::planus::alloc::boxed::Box<self::BodyCompression>,
                        >,
                        ///  Some types such as Utf8View are represented using a variable number of buffers.
                        ///  For each such Field in the pre-ordered flattened logical schema, there will be
                        ///  an entry in variadicBufferCounts to indicate the number of number of variadic
                        ///  buffers which belong to that Field in the current RecordBatch.
                        ///
                        ///  For example, the schema
                        ///      col1: Struct<alpha: Int32, beta: BinaryView, gamma: Float64>
                        ///      col2: Utf8View
                        ///  contains two Fields with variadic buffers so variadicBufferCounts will have
                        ///  two entries, the first counting the variadic buffers of `col1.beta` and the
                        ///  second counting `col2`'s.
                        ///
                        ///  This field may be omitted if and only if the schema contains no Fields with
                        ///  a variable number of buffers, such as BinaryView and Utf8View.
                        pub variadic_buffer_counts:
                            ::core::option::Option<::planus::alloc::vec::Vec<i64>>,
                    }

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for RecordBatch {
                        fn default() -> Self {
                            Self {
                                length: 0,
                                nodes: ::core::default::Default::default(),
                                buffers: ::core::default::Default::default(),
                                compression: ::core::default::Default::default(),
                                variadic_buffer_counts: ::core::default::Default::default(),
                            }
                        }
                    }

                    impl RecordBatch {
                        /// Creates a [RecordBatchBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> RecordBatchBuilder<()> {
                            RecordBatchBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            field_length: impl ::planus::WriteAsDefault<i64, i64>,
                            field_nodes: impl ::planus::WriteAsOptional<
                                ::planus::Offset<[self::FieldNode]>,
                            >,
                            field_buffers: impl ::planus::WriteAsOptional<
                                ::planus::Offset<[self::Buffer]>,
                            >,
                            field_compression: impl ::planus::WriteAsOptional<
                                ::planus::Offset<self::BodyCompression>,
                            >,
                            field_variadic_buffer_counts: impl ::planus::WriteAsOptional<
                                ::planus::Offset<[i64]>,
                            >,
                        ) -> ::planus::Offset<Self> {
                            let prepared_length = field_length.prepare(builder, &0);
                            let prepared_nodes = field_nodes.prepare(builder);
                            let prepared_buffers = field_buffers.prepare(builder);
                            let prepared_compression = field_compression.prepare(builder);
                            let prepared_variadic_buffer_counts =
                                field_variadic_buffer_counts.prepare(builder);

                            let mut table_writer: ::planus::table_writer::TableWriter<14> =
                                ::core::default::Default::default();
                            if prepared_length.is_some() {
                                table_writer.write_entry::<i64>(0);
                            }
                            if prepared_nodes.is_some() {
                                table_writer.write_entry::<::planus::Offset<[self::FieldNode]>>(1);
                            }
                            if prepared_buffers.is_some() {
                                table_writer.write_entry::<::planus::Offset<[self::Buffer]>>(2);
                            }
                            if prepared_compression.is_some() {
                                table_writer
                                    .write_entry::<::planus::Offset<self::BodyCompression>>(3);
                            }
                            if prepared_variadic_buffer_counts.is_some() {
                                table_writer.write_entry::<::planus::Offset<[i64]>>(4);
                            }

                            unsafe {
                                table_writer.finish(builder, |object_writer| {
                                    if let ::core::option::Option::Some(prepared_length) =
                                        prepared_length
                                    {
                                        object_writer.write::<_, _, 8>(&prepared_length);
                                    }
                                    if let ::core::option::Option::Some(prepared_nodes) =
                                        prepared_nodes
                                    {
                                        object_writer.write::<_, _, 4>(&prepared_nodes);
                                    }
                                    if let ::core::option::Option::Some(prepared_buffers) =
                                        prepared_buffers
                                    {
                                        object_writer.write::<_, _, 4>(&prepared_buffers);
                                    }
                                    if let ::core::option::Option::Some(prepared_compression) =
                                        prepared_compression
                                    {
                                        object_writer.write::<_, _, 4>(&prepared_compression);
                                    }
                                    if let ::core::option::Option::Some(
                                        prepared_variadic_buffer_counts,
                                    ) = prepared_variadic_buffer_counts
                                    {
                                        object_writer
                                            .write::<_, _, 4>(&prepared_variadic_buffer_counts);
                                    }
                                });
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<RecordBatch>> for RecordBatch {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<RecordBatch> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<RecordBatch>> for RecordBatch {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<RecordBatch>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<RecordBatch> for RecordBatch {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<RecordBatch> {
                            RecordBatch::create(
                                builder,
                                self.length,
                                &self.nodes,
                                &self.buffers,
                                &self.compression,
                                &self.variadic_buffer_counts,
                            )
                        }
                    }

                    /// Builder for serializing an instance of the [RecordBatch] type.
                    ///
                    /// Can be created using the [RecordBatch::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct RecordBatchBuilder<State>(State);

                    impl RecordBatchBuilder<()> {
                        /// Setter for the [`length` field](RecordBatch#structfield.length).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn length<T0>(self, value: T0) -> RecordBatchBuilder<(T0,)>
                        where
                            T0: ::planus::WriteAsDefault<i64, i64>,
                        {
                            RecordBatchBuilder((value,))
                        }

                        /// Sets the [`length` field](RecordBatch#structfield.length) to the default value.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn length_as_default(
                            self,
                        ) -> RecordBatchBuilder<(::planus::DefaultValue,)> {
                            self.length(::planus::DefaultValue)
                        }
                    }

                    impl<T0> RecordBatchBuilder<(T0,)> {
                        /// Setter for the [`nodes` field](RecordBatch#structfield.nodes).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn nodes<T1>(self, value: T1) -> RecordBatchBuilder<(T0, T1)>
                        where
                            T1: ::planus::WriteAsOptional<::planus::Offset<[self::FieldNode]>>,
                        {
                            let (v0,) = self.0;
                            RecordBatchBuilder((v0, value))
                        }

                        /// Sets the [`nodes` field](RecordBatch#structfield.nodes) to null.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn nodes_as_null(self) -> RecordBatchBuilder<(T0, ())> {
                            self.nodes(())
                        }
                    }

                    impl<T0, T1> RecordBatchBuilder<(T0, T1)> {
                        /// Setter for the [`buffers` field](RecordBatch#structfield.buffers).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn buffers<T2>(self, value: T2) -> RecordBatchBuilder<(T0, T1, T2)>
                        where
                            T2: ::planus::WriteAsOptional<::planus::Offset<[self::Buffer]>>,
                        {
                            let (v0, v1) = self.0;
                            RecordBatchBuilder((v0, v1, value))
                        }

                        /// Sets the [`buffers` field](RecordBatch#structfield.buffers) to null.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn buffers_as_null(self) -> RecordBatchBuilder<(T0, T1, ())> {
                            self.buffers(())
                        }
                    }

                    impl<T0, T1, T2> RecordBatchBuilder<(T0, T1, T2)> {
                        /// Setter for the [`compression` field](RecordBatch#structfield.compression).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn compression<T3>(
                            self,
                            value: T3,
                        ) -> RecordBatchBuilder<(T0, T1, T2, T3)>
                        where
                            T3: ::planus::WriteAsOptional<::planus::Offset<self::BodyCompression>>,
                        {
                            let (v0, v1, v2) = self.0;
                            RecordBatchBuilder((v0, v1, v2, value))
                        }

                        /// Sets the [`compression` field](RecordBatch#structfield.compression) to null.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn compression_as_null(self) -> RecordBatchBuilder<(T0, T1, T2, ())> {
                            self.compression(())
                        }
                    }

                    impl<T0, T1, T2, T3> RecordBatchBuilder<(T0, T1, T2, T3)> {
                        /// Setter for the [`variadicBufferCounts` field](RecordBatch#structfield.variadic_buffer_counts).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn variadic_buffer_counts<T4>(
                            self,
                            value: T4,
                        ) -> RecordBatchBuilder<(T0, T1, T2, T3, T4)>
                        where
                            T4: ::planus::WriteAsOptional<::planus::Offset<[i64]>>,
                        {
                            let (v0, v1, v2, v3) = self.0;
                            RecordBatchBuilder((v0, v1, v2, v3, value))
                        }

                        /// Sets the [`variadicBufferCounts` field](RecordBatch#structfield.variadic_buffer_counts) to null.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn variadic_buffer_counts_as_null(
                            self,
                        ) -> RecordBatchBuilder<(T0, T1, T2, T3, ())> {
                            self.variadic_buffer_counts(())
                        }
                    }

                    impl<T0, T1, T2, T3, T4> RecordBatchBuilder<(T0, T1, T2, T3, T4)> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [RecordBatch].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<RecordBatch>
                        where
                            Self: ::planus::WriteAsOffset<RecordBatch>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<i64, i64>,
                            T1: ::planus::WriteAsOptional<::planus::Offset<[self::FieldNode]>>,
                            T2: ::planus::WriteAsOptional<::planus::Offset<[self::Buffer]>>,
                            T3: ::planus::WriteAsOptional<::planus::Offset<self::BodyCompression>>,
                            T4: ::planus::WriteAsOptional<::planus::Offset<[i64]>>,
                        > ::planus::WriteAs<::planus::Offset<RecordBatch>>
                        for RecordBatchBuilder<(T0, T1, T2, T3, T4)>
                    {
                        type Prepared = ::planus::Offset<RecordBatch>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<RecordBatch> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<i64, i64>,
                            T1: ::planus::WriteAsOptional<::planus::Offset<[self::FieldNode]>>,
                            T2: ::planus::WriteAsOptional<::planus::Offset<[self::Buffer]>>,
                            T3: ::planus::WriteAsOptional<::planus::Offset<self::BodyCompression>>,
                            T4: ::planus::WriteAsOptional<::planus::Offset<[i64]>>,
                        >
                        ::planus::WriteAsOptional<::planus::Offset<RecordBatch>>
                        for RecordBatchBuilder<(T0, T1, T2, T3, T4)>
                    {
                        type Prepared = ::planus::Offset<RecordBatch>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<RecordBatch>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<i64, i64>,
                            T1: ::planus::WriteAsOptional<::planus::Offset<[self::FieldNode]>>,
                            T2: ::planus::WriteAsOptional<::planus::Offset<[self::Buffer]>>,
                            T3: ::planus::WriteAsOptional<::planus::Offset<self::BodyCompression>>,
                            T4: ::planus::WriteAsOptional<::planus::Offset<[i64]>>,
                        > ::planus::WriteAsOffset<RecordBatch>
                        for RecordBatchBuilder<(T0, T1, T2, T3, T4)>
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<RecordBatch> {
                            let (v0, v1, v2, v3, v4) = &self.0;
                            RecordBatch::create(builder, v0, v1, v2, v3, v4)
                        }
                    }

                    /// Reference to a deserialized [RecordBatch].
                    #[derive(Copy, Clone)]
                    pub struct RecordBatchRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> RecordBatchRef<'a> {
                        /// Getter for the [`length` field](RecordBatch#structfield.length).
                        #[inline]
                        pub fn length(&self) -> ::planus::Result<i64> {
                            ::core::result::Result::Ok(
                                self.0.access(0, "RecordBatch", "length")?.unwrap_or(0),
                            )
                        }

                        /// Getter for the [`nodes` field](RecordBatch#structfield.nodes).
                        #[inline]
                        pub fn nodes(
                            &self,
                        ) -> ::planus::Result<
                            ::core::option::Option<::planus::Vector<'a, self::FieldNodeRef<'a>>>,
                        > {
                            self.0.access(1, "RecordBatch", "nodes")
                        }

                        /// Getter for the [`buffers` field](RecordBatch#structfield.buffers).
                        #[inline]
                        pub fn buffers(
                            &self,
                        ) -> ::planus::Result<
                            ::core::option::Option<::planus::Vector<'a, self::BufferRef<'a>>>,
                        > {
                            self.0.access(2, "RecordBatch", "buffers")
                        }

                        /// Getter for the [`compression` field](RecordBatch#structfield.compression).
                        #[inline]
                        pub fn compression(
                            &self,
                        ) -> ::planus::Result<::core::option::Option<self::BodyCompressionRef<'a>>>
                        {
                            self.0.access(3, "RecordBatch", "compression")
                        }

                        /// Getter for the [`variadicBufferCounts` field](RecordBatch#structfield.variadic_buffer_counts).
                        #[inline]
                        pub fn variadic_buffer_counts(
                            &self,
                        ) -> ::planus::Result<::core::option::Option<::planus::Vector<'a, i64>>>
                        {
                            self.0.access(4, "RecordBatch", "variadic_buffer_counts")
                        }
                    }

                    impl<'a> ::core::fmt::Debug for RecordBatchRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("RecordBatchRef");
                            f.field("length", &self.length());
                            if let ::core::option::Option::Some(field_nodes) =
                                self.nodes().transpose()
                            {
                                f.field("nodes", &field_nodes);
                            }
                            if let ::core::option::Option::Some(field_buffers) =
                                self.buffers().transpose()
                            {
                                f.field("buffers", &field_buffers);
                            }
                            if let ::core::option::Option::Some(field_compression) =
                                self.compression().transpose()
                            {
                                f.field("compression", &field_compression);
                            }
                            if let ::core::option::Option::Some(field_variadic_buffer_counts) =
                                self.variadic_buffer_counts().transpose()
                            {
                                f.field("variadic_buffer_counts", &field_variadic_buffer_counts);
                            }
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<RecordBatchRef<'a>> for RecordBatch {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: RecordBatchRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                length: ::core::convert::TryInto::try_into(value.length()?)?,
                                nodes: if let ::core::option::Option::Some(nodes) = value.nodes()? {
                                    ::core::option::Option::Some(nodes.to_vec()?)
                                } else {
                                    ::core::option::Option::None
                                },
                                buffers: if let ::core::option::Option::Some(buffers) =
                                    value.buffers()?
                                {
                                    ::core::option::Option::Some(buffers.to_vec()?)
                                } else {
                                    ::core::option::Option::None
                                },
                                compression: if let ::core::option::Option::Some(compression) =
                                    value.compression()?
                                {
                                    ::core::option::Option::Some(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryInto::try_into(compression)?,
                                    ))
                                } else {
                                    ::core::option::Option::None
                                },
                                variadic_buffer_counts: if let ::core::option::Option::Some(
                                    variadic_buffer_counts,
                                ) = value.variadic_buffer_counts()?
                                {
                                    ::core::option::Option::Some(variadic_buffer_counts.to_vec()?)
                                } else {
                                    ::core::option::Option::None
                                },
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for RecordBatchRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for RecordBatchRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[RecordBatchRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<RecordBatch>> for RecordBatch {
                        type Value = ::planus::Offset<RecordBatch>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<RecordBatch>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for RecordBatchRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[RecordBatchRef]",
                                    "read_as_root",
                                    0,
                                )
                            })
                        }
                    }

                    ///  For sending dictionary encoding information. Any Field can be
                    ///  dictionary-encoded, but in this case none of its children may be
                    ///  dictionary-encoded.
                    ///  There is one vector / column per dictionary, but that vector / column
                    ///  may be spread across multiple dictionary batches by using the isDelta
                    ///  flag
                    ///
                    /// Generated from these locations:
                    /// * Table `DictionaryBatch` in the file `Message.fbs:127`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct DictionaryBatch {
                        /// The field `id` in the table `DictionaryBatch`
                        pub id: i64,
                        /// The field `data` in the table `DictionaryBatch`
                        pub data:
                            ::core::option::Option<::planus::alloc::boxed::Box<self::RecordBatch>>,
                        ///  If isDelta is true the values in the dictionary are to be appended to a
                        ///  dictionary with the indicated id. If isDelta is false this dictionary
                        ///  should replace the existing dictionary.
                        pub is_delta: bool,
                    }

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for DictionaryBatch {
                        fn default() -> Self {
                            Self {
                                id: 0,
                                data: ::core::default::Default::default(),
                                is_delta: false,
                            }
                        }
                    }

                    impl DictionaryBatch {
                        /// Creates a [DictionaryBatchBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> DictionaryBatchBuilder<()> {
                            DictionaryBatchBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            field_id: impl ::planus::WriteAsDefault<i64, i64>,
                            field_data: impl ::planus::WriteAsOptional<
                                ::planus::Offset<self::RecordBatch>,
                            >,
                            field_is_delta: impl ::planus::WriteAsDefault<bool, bool>,
                        ) -> ::planus::Offset<Self> {
                            let prepared_id = field_id.prepare(builder, &0);
                            let prepared_data = field_data.prepare(builder);
                            let prepared_is_delta = field_is_delta.prepare(builder, &false);

                            let mut table_writer: ::planus::table_writer::TableWriter<10> =
                                ::core::default::Default::default();
                            if prepared_id.is_some() {
                                table_writer.write_entry::<i64>(0);
                            }
                            if prepared_data.is_some() {
                                table_writer.write_entry::<::planus::Offset<self::RecordBatch>>(1);
                            }
                            if prepared_is_delta.is_some() {
                                table_writer.write_entry::<bool>(2);
                            }

                            unsafe {
                                table_writer.finish(builder, |object_writer| {
                                    if let ::core::option::Option::Some(prepared_id) = prepared_id {
                                        object_writer.write::<_, _, 8>(&prepared_id);
                                    }
                                    if let ::core::option::Option::Some(prepared_data) =
                                        prepared_data
                                    {
                                        object_writer.write::<_, _, 4>(&prepared_data);
                                    }
                                    if let ::core::option::Option::Some(prepared_is_delta) =
                                        prepared_is_delta
                                    {
                                        object_writer.write::<_, _, 1>(&prepared_is_delta);
                                    }
                                });
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<DictionaryBatch>> for DictionaryBatch {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<DictionaryBatch> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<DictionaryBatch>> for DictionaryBatch {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<DictionaryBatch>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<DictionaryBatch> for DictionaryBatch {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<DictionaryBatch> {
                            DictionaryBatch::create(builder, self.id, &self.data, self.is_delta)
                        }
                    }

                    /// Builder for serializing an instance of the [DictionaryBatch] type.
                    ///
                    /// Can be created using the [DictionaryBatch::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct DictionaryBatchBuilder<State>(State);

                    impl DictionaryBatchBuilder<()> {
                        /// Setter for the [`id` field](DictionaryBatch#structfield.id).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn id<T0>(self, value: T0) -> DictionaryBatchBuilder<(T0,)>
                        where
                            T0: ::planus::WriteAsDefault<i64, i64>,
                        {
                            DictionaryBatchBuilder((value,))
                        }

                        /// Sets the [`id` field](DictionaryBatch#structfield.id) to the default value.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn id_as_default(
                            self,
                        ) -> DictionaryBatchBuilder<(::planus::DefaultValue,)>
                        {
                            self.id(::planus::DefaultValue)
                        }
                    }

                    impl<T0> DictionaryBatchBuilder<(T0,)> {
                        /// Setter for the [`data` field](DictionaryBatch#structfield.data).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn data<T1>(self, value: T1) -> DictionaryBatchBuilder<(T0, T1)>
                        where
                            T1: ::planus::WriteAsOptional<::planus::Offset<self::RecordBatch>>,
                        {
                            let (v0,) = self.0;
                            DictionaryBatchBuilder((v0, value))
                        }

                        /// Sets the [`data` field](DictionaryBatch#structfield.data) to null.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn data_as_null(self) -> DictionaryBatchBuilder<(T0, ())> {
                            self.data(())
                        }
                    }

                    impl<T0, T1> DictionaryBatchBuilder<(T0, T1)> {
                        /// Setter for the [`isDelta` field](DictionaryBatch#structfield.is_delta).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn is_delta<T2>(self, value: T2) -> DictionaryBatchBuilder<(T0, T1, T2)>
                        where
                            T2: ::planus::WriteAsDefault<bool, bool>,
                        {
                            let (v0, v1) = self.0;
                            DictionaryBatchBuilder((v0, v1, value))
                        }

                        /// Sets the [`isDelta` field](DictionaryBatch#structfield.is_delta) to the default value.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn is_delta_as_default(
                            self,
                        ) -> DictionaryBatchBuilder<(T0, T1, ::planus::DefaultValue)>
                        {
                            self.is_delta(::planus::DefaultValue)
                        }
                    }

                    impl<T0, T1, T2> DictionaryBatchBuilder<(T0, T1, T2)> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [DictionaryBatch].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<DictionaryBatch>
                        where
                            Self: ::planus::WriteAsOffset<DictionaryBatch>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<i64, i64>,
                            T1: ::planus::WriteAsOptional<::planus::Offset<self::RecordBatch>>,
                            T2: ::planus::WriteAsDefault<bool, bool>,
                        > ::planus::WriteAs<::planus::Offset<DictionaryBatch>>
                        for DictionaryBatchBuilder<(T0, T1, T2)>
                    {
                        type Prepared = ::planus::Offset<DictionaryBatch>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<DictionaryBatch> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<i64, i64>,
                            T1: ::planus::WriteAsOptional<::planus::Offset<self::RecordBatch>>,
                            T2: ::planus::WriteAsDefault<bool, bool>,
                        >
                        ::planus::WriteAsOptional<::planus::Offset<DictionaryBatch>>
                        for DictionaryBatchBuilder<(T0, T1, T2)>
                    {
                        type Prepared = ::planus::Offset<DictionaryBatch>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<DictionaryBatch>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<i64, i64>,
                            T1: ::planus::WriteAsOptional<::planus::Offset<self::RecordBatch>>,
                            T2: ::planus::WriteAsDefault<bool, bool>,
                        > ::planus::WriteAsOffset<DictionaryBatch>
                        for DictionaryBatchBuilder<(T0, T1, T2)>
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<DictionaryBatch> {
                            let (v0, v1, v2) = &self.0;
                            DictionaryBatch::create(builder, v0, v1, v2)
                        }
                    }

                    /// Reference to a deserialized [DictionaryBatch].
                    #[derive(Copy, Clone)]
                    pub struct DictionaryBatchRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> DictionaryBatchRef<'a> {
                        /// Getter for the [`id` field](DictionaryBatch#structfield.id).
                        #[inline]
                        pub fn id(&self) -> ::planus::Result<i64> {
                            ::core::result::Result::Ok(
                                self.0.access(0, "DictionaryBatch", "id")?.unwrap_or(0),
                            )
                        }

                        /// Getter for the [`data` field](DictionaryBatch#structfield.data).
                        #[inline]
                        pub fn data(
                            &self,
                        ) -> ::planus::Result<::core::option::Option<self::RecordBatchRef<'a>>>
                        {
                            self.0.access(1, "DictionaryBatch", "data")
                        }

                        /// Getter for the [`isDelta` field](DictionaryBatch#structfield.is_delta).
                        #[inline]
                        pub fn is_delta(&self) -> ::planus::Result<bool> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(2, "DictionaryBatch", "is_delta")?
                                    .unwrap_or(false),
                            )
                        }
                    }

                    impl<'a> ::core::fmt::Debug for DictionaryBatchRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("DictionaryBatchRef");
                            f.field("id", &self.id());
                            if let ::core::option::Option::Some(field_data) =
                                self.data().transpose()
                            {
                                f.field("data", &field_data);
                            }
                            f.field("is_delta", &self.is_delta());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<DictionaryBatchRef<'a>> for DictionaryBatch {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: DictionaryBatchRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                id: ::core::convert::TryInto::try_into(value.id()?)?,
                                data: if let ::core::option::Option::Some(data) = value.data()? {
                                    ::core::option::Option::Some(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryInto::try_into(data)?,
                                    ))
                                } else {
                                    ::core::option::Option::None
                                },
                                is_delta: ::core::convert::TryInto::try_into(value.is_delta()?)?,
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for DictionaryBatchRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for DictionaryBatchRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[DictionaryBatchRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<DictionaryBatch>> for DictionaryBatch {
                        type Value = ::planus::Offset<DictionaryBatch>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<DictionaryBatch>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for DictionaryBatchRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[DictionaryBatchRef]",
                                    "read_as_root",
                                    0,
                                )
                            })
                        }
                    }

                    ///  ----------------------------------------------------------------------
                    ///  The root Message type
                    ///  This union enables us to easily send different message types without
                    ///  redundant storage, and in the future we can easily add new message types.
                    ///
                    ///  Arrow implementations do not need to implement all of the message types,
                    ///  which may include experimental metadata types. For maximum compatibility,
                    ///  it is best to send data using RecordBatch
                    ///
                    /// Generated from these locations:
                    /// * Union `MessageHeader` in the file `Message.fbs:146`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub enum MessageHeader {
                        /// The variant of type `Schema` in the union `MessageHeader`
                        Schema(::planus::alloc::boxed::Box<self::Schema>),

                        /// The variant of type `DictionaryBatch` in the union `MessageHeader`
                        DictionaryBatch(::planus::alloc::boxed::Box<self::DictionaryBatch>),

                        /// The variant of type `RecordBatch` in the union `MessageHeader`
                        RecordBatch(::planus::alloc::boxed::Box<self::RecordBatch>),

                        /// The variant of type `Tensor` in the union `MessageHeader`
                        Tensor(::planus::alloc::boxed::Box<self::Tensor>),

                        /// The variant of type `SparseTensor` in the union `MessageHeader`
                        SparseTensor(::planus::alloc::boxed::Box<self::SparseTensor>),
                    }

                    impl MessageHeader {
                        /// Creates a [MessageHeaderBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> MessageHeaderBuilder<::planus::Uninitialized> {
                            MessageHeaderBuilder(::planus::Uninitialized)
                        }

                        #[inline]
                        pub fn create_schema(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::Schema>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(1, value.prepare(builder).downcast())
                        }

                        #[inline]
                        pub fn create_dictionary_batch(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::DictionaryBatch>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(2, value.prepare(builder).downcast())
                        }

                        #[inline]
                        pub fn create_record_batch(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::RecordBatch>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(3, value.prepare(builder).downcast())
                        }

                        #[inline]
                        pub fn create_tensor(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::Tensor>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(4, value.prepare(builder).downcast())
                        }

                        #[inline]
                        pub fn create_sparse_tensor(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::SparseTensor>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(5, value.prepare(builder).downcast())
                        }
                    }

                    impl ::planus::WriteAsUnion<MessageHeader> for MessageHeader {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<Self> {
                            match self {
                                Self::Schema(value) => Self::create_schema(builder, value),
                                Self::DictionaryBatch(value) => {
                                    Self::create_dictionary_batch(builder, value)
                                }
                                Self::RecordBatch(value) => {
                                    Self::create_record_batch(builder, value)
                                }
                                Self::Tensor(value) => Self::create_tensor(builder, value),
                                Self::SparseTensor(value) => {
                                    Self::create_sparse_tensor(builder, value)
                                }
                            }
                        }
                    }

                    impl ::planus::WriteAsOptionalUnion<MessageHeader> for MessageHeader {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<Self>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }

                    /// Builder for serializing an instance of the [MessageHeader] type.
                    ///
                    /// Can be created using the [MessageHeader::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct MessageHeaderBuilder<T>(T);

                    impl MessageHeaderBuilder<::planus::Uninitialized> {
                        /// Creates an instance of the [`Schema` variant](MessageHeader#variant.Schema).
                        #[inline]
                        pub fn schema<T>(
                            self,
                            value: T,
                        ) -> MessageHeaderBuilder<::planus::Initialized<1, T>>
                        where
                            T: ::planus::WriteAsOffset<self::Schema>,
                        {
                            MessageHeaderBuilder(::planus::Initialized(value))
                        }

                        /// Creates an instance of the [`DictionaryBatch` variant](MessageHeader#variant.DictionaryBatch).
                        #[inline]
                        pub fn dictionary_batch<T>(
                            self,
                            value: T,
                        ) -> MessageHeaderBuilder<::planus::Initialized<2, T>>
                        where
                            T: ::planus::WriteAsOffset<self::DictionaryBatch>,
                        {
                            MessageHeaderBuilder(::planus::Initialized(value))
                        }

                        /// Creates an instance of the [`RecordBatch` variant](MessageHeader#variant.RecordBatch).
                        #[inline]
                        pub fn record_batch<T>(
                            self,
                            value: T,
                        ) -> MessageHeaderBuilder<::planus::Initialized<3, T>>
                        where
                            T: ::planus::WriteAsOffset<self::RecordBatch>,
                        {
                            MessageHeaderBuilder(::planus::Initialized(value))
                        }

                        /// Creates an instance of the [`Tensor` variant](MessageHeader#variant.Tensor).
                        #[inline]
                        pub fn tensor<T>(
                            self,
                            value: T,
                        ) -> MessageHeaderBuilder<::planus::Initialized<4, T>>
                        where
                            T: ::planus::WriteAsOffset<self::Tensor>,
                        {
                            MessageHeaderBuilder(::planus::Initialized(value))
                        }

                        /// Creates an instance of the [`SparseTensor` variant](MessageHeader#variant.SparseTensor).
                        #[inline]
                        pub fn sparse_tensor<T>(
                            self,
                            value: T,
                        ) -> MessageHeaderBuilder<::planus::Initialized<5, T>>
                        where
                            T: ::planus::WriteAsOffset<self::SparseTensor>,
                        {
                            MessageHeaderBuilder(::planus::Initialized(value))
                        }
                    }

                    impl<const N: u8, T> MessageHeaderBuilder<::planus::Initialized<N, T>> {
                        /// Finish writing the builder to get an [UnionOffset](::planus::UnionOffset) to a serialized [MessageHeader].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<MessageHeader>
                        where
                            Self: ::planus::WriteAsUnion<MessageHeader>,
                        {
                            ::planus::WriteAsUnion::prepare(&self, builder)
                        }
                    }

                    impl<T> ::planus::WriteAsUnion<MessageHeader> for MessageHeaderBuilder<::planus::Initialized<1, T>>
                    where
                        T: ::planus::WriteAsOffset<self::Schema>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<MessageHeader> {
                            ::planus::UnionOffset::new(1, (self.0).0.prepare(builder).downcast())
                        }
                    }

                    impl<T> ::planus::WriteAsOptionalUnion<MessageHeader>
                        for MessageHeaderBuilder<::planus::Initialized<1, T>>
                    where
                        T: ::planus::WriteAsOffset<self::Schema>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<MessageHeader>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }
                    impl<T> ::planus::WriteAsUnion<MessageHeader> for MessageHeaderBuilder<::planus::Initialized<2, T>>
                    where
                        T: ::planus::WriteAsOffset<self::DictionaryBatch>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<MessageHeader> {
                            ::planus::UnionOffset::new(2, (self.0).0.prepare(builder).downcast())
                        }
                    }

                    impl<T> ::planus::WriteAsOptionalUnion<MessageHeader>
                        for MessageHeaderBuilder<::planus::Initialized<2, T>>
                    where
                        T: ::planus::WriteAsOffset<self::DictionaryBatch>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<MessageHeader>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }
                    impl<T> ::planus::WriteAsUnion<MessageHeader> for MessageHeaderBuilder<::planus::Initialized<3, T>>
                    where
                        T: ::planus::WriteAsOffset<self::RecordBatch>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<MessageHeader> {
                            ::planus::UnionOffset::new(3, (self.0).0.prepare(builder).downcast())
                        }
                    }

                    impl<T> ::planus::WriteAsOptionalUnion<MessageHeader>
                        for MessageHeaderBuilder<::planus::Initialized<3, T>>
                    where
                        T: ::planus::WriteAsOffset<self::RecordBatch>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<MessageHeader>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }
                    impl<T> ::planus::WriteAsUnion<MessageHeader> for MessageHeaderBuilder<::planus::Initialized<4, T>>
                    where
                        T: ::planus::WriteAsOffset<self::Tensor>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<MessageHeader> {
                            ::planus::UnionOffset::new(4, (self.0).0.prepare(builder).downcast())
                        }
                    }

                    impl<T> ::planus::WriteAsOptionalUnion<MessageHeader>
                        for MessageHeaderBuilder<::planus::Initialized<4, T>>
                    where
                        T: ::planus::WriteAsOffset<self::Tensor>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<MessageHeader>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }
                    impl<T> ::planus::WriteAsUnion<MessageHeader> for MessageHeaderBuilder<::planus::Initialized<5, T>>
                    where
                        T: ::planus::WriteAsOffset<self::SparseTensor>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<MessageHeader> {
                            ::planus::UnionOffset::new(5, (self.0).0.prepare(builder).downcast())
                        }
                    }

                    impl<T> ::planus::WriteAsOptionalUnion<MessageHeader>
                        for MessageHeaderBuilder<::planus::Initialized<5, T>>
                    where
                        T: ::planus::WriteAsOffset<self::SparseTensor>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<MessageHeader>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }

                    /// Reference to a deserialized [MessageHeader].
                    #[derive(Copy, Clone, Debug)]
                    pub enum MessageHeaderRef<'a> {
                        Schema(self::SchemaRef<'a>),
                        DictionaryBatch(self::DictionaryBatchRef<'a>),
                        RecordBatch(self::RecordBatchRef<'a>),
                        Tensor(self::TensorRef<'a>),
                        SparseTensor(self::SparseTensorRef<'a>),
                    }

                    impl<'a> ::core::convert::TryFrom<MessageHeaderRef<'a>> for MessageHeader {
                        type Error = ::planus::Error;

                        fn try_from(value: MessageHeaderRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(match value {
                                MessageHeaderRef::Schema(value) => {
                                    Self::Schema(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                MessageHeaderRef::DictionaryBatch(value) => {
                                    Self::DictionaryBatch(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                MessageHeaderRef::RecordBatch(value) => {
                                    Self::RecordBatch(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                MessageHeaderRef::Tensor(value) => {
                                    Self::Tensor(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                MessageHeaderRef::SparseTensor(value) => {
                                    Self::SparseTensor(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }
                            })
                        }
                    }

                    impl<'a> ::planus::TableReadUnion<'a> for MessageHeaderRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            tag: u8,
                            field_offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            match tag {
                                1 => ::core::result::Result::Ok(Self::Schema(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                2 => ::core::result::Result::Ok(Self::DictionaryBatch(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                3 => ::core::result::Result::Ok(Self::RecordBatch(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                4 => ::core::result::Result::Ok(Self::Tensor(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                5 => ::core::result::Result::Ok(Self::SparseTensor(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                _ => ::core::result::Result::Err(
                                    ::planus::errors::ErrorKind::UnknownUnionTag { tag },
                                ),
                            }
                        }
                    }

                    impl<'a> ::planus::VectorReadUnion<'a> for MessageHeaderRef<'a> {
                        const VECTOR_NAME: &'static str = "[MessageHeaderRef]";
                    }

                    /// The table `Message` in the namespace `org.apache.arrow.flatbuf`
                    ///
                    /// Generated from these locations:
                    /// * Table `Message` in the file `Message.fbs:150`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct Message {
                        /// The field `version` in the table `Message`
                        pub version: self::MetadataVersion,
                        /// The field `header` in the table `Message`
                        pub header: ::core::option::Option<self::MessageHeader>,
                        /// The field `bodyLength` in the table `Message`
                        pub body_length: i64,
                        /// The field `custom_metadata` in the table `Message`
                        pub custom_metadata:
                            ::core::option::Option<::planus::alloc::vec::Vec<self::KeyValue>>,
                    }

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for Message {
                        fn default() -> Self {
                            Self {
                                version: self::MetadataVersion::V1,
                                header: ::core::default::Default::default(),
                                body_length: 0,
                                custom_metadata: ::core::default::Default::default(),
                            }
                        }
                    }

                    impl Message {
                        /// Creates a [MessageBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> MessageBuilder<()> {
                            MessageBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            field_version: impl ::planus::WriteAsDefault<
                                self::MetadataVersion,
                                self::MetadataVersion,
                            >,
                            field_header: impl ::planus::WriteAsOptionalUnion<self::MessageHeader>,
                            field_body_length: impl ::planus::WriteAsDefault<i64, i64>,
                            field_custom_metadata: impl ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::KeyValue>]>,
                            >,
                        ) -> ::planus::Offset<Self> {
                            let prepared_version =
                                field_version.prepare(builder, &self::MetadataVersion::V1);
                            let prepared_header = field_header.prepare(builder);
                            let prepared_body_length = field_body_length.prepare(builder, &0);
                            let prepared_custom_metadata = field_custom_metadata.prepare(builder);

                            let mut table_writer: ::planus::table_writer::TableWriter<14> =
                                ::core::default::Default::default();
                            if prepared_body_length.is_some() {
                                table_writer.write_entry::<i64>(3);
                            }
                            if prepared_header.is_some() {
                                table_writer
                                    .write_entry::<::planus::Offset<self::MessageHeader>>(2);
                            }
                            if prepared_custom_metadata.is_some() {
                                table_writer.write_entry::<::planus::Offset<[::planus::Offset<self::KeyValue>]>>(4);
                            }
                            if prepared_version.is_some() {
                                table_writer.write_entry::<self::MetadataVersion>(0);
                            }
                            if prepared_header.is_some() {
                                table_writer.write_entry::<u8>(1);
                            }

                            unsafe {
                                table_writer.finish(builder, |object_writer| {
                                    if let ::core::option::Option::Some(prepared_body_length) =
                                        prepared_body_length
                                    {
                                        object_writer.write::<_, _, 8>(&prepared_body_length);
                                    }
                                    if let ::core::option::Option::Some(prepared_header) =
                                        prepared_header
                                    {
                                        object_writer.write::<_, _, 4>(&prepared_header.offset());
                                    }
                                    if let ::core::option::Option::Some(prepared_custom_metadata) =
                                        prepared_custom_metadata
                                    {
                                        object_writer.write::<_, _, 4>(&prepared_custom_metadata);
                                    }
                                    if let ::core::option::Option::Some(prepared_version) =
                                        prepared_version
                                    {
                                        object_writer.write::<_, _, 2>(&prepared_version);
                                    }
                                    if let ::core::option::Option::Some(prepared_header) =
                                        prepared_header
                                    {
                                        object_writer.write::<_, _, 1>(&prepared_header.tag());
                                    }
                                });
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Message>> for Message {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Message> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Message>> for Message {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Message>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Message> for Message {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Message> {
                            Message::create(
                                builder,
                                self.version,
                                &self.header,
                                self.body_length,
                                &self.custom_metadata,
                            )
                        }
                    }

                    /// Builder for serializing an instance of the [Message] type.
                    ///
                    /// Can be created using the [Message::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct MessageBuilder<State>(State);

                    impl MessageBuilder<()> {
                        /// Setter for the [`version` field](Message#structfield.version).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn version<T0>(self, value: T0) -> MessageBuilder<(T0,)>
                        where
                            T0: ::planus::WriteAsDefault<
                                self::MetadataVersion,
                                self::MetadataVersion,
                            >,
                        {
                            MessageBuilder((value,))
                        }

                        /// Sets the [`version` field](Message#structfield.version) to the default value.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn version_as_default(
                            self,
                        ) -> MessageBuilder<(::planus::DefaultValue,)> {
                            self.version(::planus::DefaultValue)
                        }
                    }

                    impl<T0> MessageBuilder<(T0,)> {
                        /// Setter for the [`header` field](Message#structfield.header).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn header<T1>(self, value: T1) -> MessageBuilder<(T0, T1)>
                        where
                            T1: ::planus::WriteAsOptionalUnion<self::MessageHeader>,
                        {
                            let (v0,) = self.0;
                            MessageBuilder((v0, value))
                        }

                        /// Sets the [`header` field](Message#structfield.header) to null.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn header_as_null(self) -> MessageBuilder<(T0, ())> {
                            self.header(())
                        }
                    }

                    impl<T0, T1> MessageBuilder<(T0, T1)> {
                        /// Setter for the [`bodyLength` field](Message#structfield.body_length).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn body_length<T2>(self, value: T2) -> MessageBuilder<(T0, T1, T2)>
                        where
                            T2: ::planus::WriteAsDefault<i64, i64>,
                        {
                            let (v0, v1) = self.0;
                            MessageBuilder((v0, v1, value))
                        }

                        /// Sets the [`bodyLength` field](Message#structfield.body_length) to the default value.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn body_length_as_default(
                            self,
                        ) -> MessageBuilder<(T0, T1, ::planus::DefaultValue)>
                        {
                            self.body_length(::planus::DefaultValue)
                        }
                    }

                    impl<T0, T1, T2> MessageBuilder<(T0, T1, T2)> {
                        /// Setter for the [`custom_metadata` field](Message#structfield.custom_metadata).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn custom_metadata<T3>(
                            self,
                            value: T3,
                        ) -> MessageBuilder<(T0, T1, T2, T3)>
                        where
                            T3: ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::KeyValue>]>,
                            >,
                        {
                            let (v0, v1, v2) = self.0;
                            MessageBuilder((v0, v1, v2, value))
                        }

                        /// Sets the [`custom_metadata` field](Message#structfield.custom_metadata) to null.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn custom_metadata_as_null(self) -> MessageBuilder<(T0, T1, T2, ())> {
                            self.custom_metadata(())
                        }
                    }

                    impl<T0, T1, T2, T3> MessageBuilder<(T0, T1, T2, T3)> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Message].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Message>
                        where
                            Self: ::planus::WriteAsOffset<Message>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<self::MetadataVersion, self::MetadataVersion>,
                            T1: ::planus::WriteAsOptionalUnion<self::MessageHeader>,
                            T2: ::planus::WriteAsDefault<i64, i64>,
                            T3: ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::KeyValue>]>,
                            >,
                        > ::planus::WriteAs<::planus::Offset<Message>>
                        for MessageBuilder<(T0, T1, T2, T3)>
                    {
                        type Prepared = ::planus::Offset<Message>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Message> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<self::MetadataVersion, self::MetadataVersion>,
                            T1: ::planus::WriteAsOptionalUnion<self::MessageHeader>,
                            T2: ::planus::WriteAsDefault<i64, i64>,
                            T3: ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::KeyValue>]>,
                            >,
                        > ::planus::WriteAsOptional<::planus::Offset<Message>>
                        for MessageBuilder<(T0, T1, T2, T3)>
                    {
                        type Prepared = ::planus::Offset<Message>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Message>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<self::MetadataVersion, self::MetadataVersion>,
                            T1: ::planus::WriteAsOptionalUnion<self::MessageHeader>,
                            T2: ::planus::WriteAsDefault<i64, i64>,
                            T3: ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::KeyValue>]>,
                            >,
                        > ::planus::WriteAsOffset<Message> for MessageBuilder<(T0, T1, T2, T3)>
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Message> {
                            let (v0, v1, v2, v3) = &self.0;
                            Message::create(builder, v0, v1, v2, v3)
                        }
                    }

                    /// Reference to a deserialized [Message].
                    #[derive(Copy, Clone)]
                    pub struct MessageRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> MessageRef<'a> {
                        /// Getter for the [`version` field](Message#structfield.version).
                        #[inline]
                        pub fn version(&self) -> ::planus::Result<self::MetadataVersion> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(0, "Message", "version")?
                                    .unwrap_or(self::MetadataVersion::V1),
                            )
                        }

                        /// Getter for the [`header` field](Message#structfield.header).
                        #[inline]
                        pub fn header(
                            &self,
                        ) -> ::planus::Result<::core::option::Option<self::MessageHeaderRef<'a>>>
                        {
                            self.0.access_union(1, "Message", "header")
                        }

                        /// Getter for the [`bodyLength` field](Message#structfield.body_length).
                        #[inline]
                        pub fn body_length(&self) -> ::planus::Result<i64> {
                            ::core::result::Result::Ok(
                                self.0.access(3, "Message", "body_length")?.unwrap_or(0),
                            )
                        }

                        /// Getter for the [`custom_metadata` field](Message#structfield.custom_metadata).
                        #[inline]
                        pub fn custom_metadata(
                            &self,
                        ) -> ::planus::Result<
                            ::core::option::Option<
                                ::planus::Vector<'a, ::planus::Result<self::KeyValueRef<'a>>>,
                            >,
                        > {
                            self.0.access(4, "Message", "custom_metadata")
                        }
                    }

                    impl<'a> ::core::fmt::Debug for MessageRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("MessageRef");
                            f.field("version", &self.version());
                            if let ::core::option::Option::Some(field_header) =
                                self.header().transpose()
                            {
                                f.field("header", &field_header);
                            }
                            f.field("body_length", &self.body_length());
                            if let ::core::option::Option::Some(field_custom_metadata) =
                                self.custom_metadata().transpose()
                            {
                                f.field("custom_metadata", &field_custom_metadata);
                            }
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<MessageRef<'a>> for Message {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: MessageRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                version: ::core::convert::TryInto::try_into(value.version()?)?,
                                header: if let ::core::option::Option::Some(header) =
                                    value.header()?
                                {
                                    ::core::option::Option::Some(
                                        ::core::convert::TryInto::try_into(header)?,
                                    )
                                } else {
                                    ::core::option::Option::None
                                },
                                body_length: ::core::convert::TryInto::try_into(
                                    value.body_length()?,
                                )?,
                                custom_metadata: if let ::core::option::Option::Some(
                                    custom_metadata,
                                ) = value.custom_metadata()?
                                {
                                    ::core::option::Option::Some(custom_metadata.to_vec_result()?)
                                } else {
                                    ::core::option::Option::None
                                },
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for MessageRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for MessageRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[MessageRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<Message>> for Message {
                        type Value = ::planus::Offset<Message>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Message>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for MessageRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[MessageRef]", "read_as_root", 0)
                            })
                        }
                    }

                    ///  ----------------------------------------------------------------------
                    ///  EXPERIMENTAL: Data structures for sparse tensors
                    ///  Coordinate (COO) format of sparse tensor index.
                    ///
                    ///  COO's index list are represented as a NxM matrix,
                    ///  where N is the number of non-zero values,
                    ///  and M is the number of dimensions of a sparse tensor.
                    ///
                    ///  indicesBuffer stores the location and size of the data of this indices
                    ///  matrix.  The value type and the stride of the indices matrix is
                    ///  specified in indicesType and indicesStrides fields.
                    ///
                    ///  For example, let X be a 2x3x4x5 tensor, and it has the following
                    ///  6 non-zero values:
                    ///  ```text
                    ///    X[0, 1, 2, 0] := 1
                    ///    X[1, 1, 2, 3] := 2
                    ///    X[0, 2, 1, 0] := 3
                    ///    X[0, 1, 3, 0] := 4
                    ///    X[0, 1, 2, 1] := 5
                    ///    X[1, 2, 0, 4] := 6
                    ///  ```
                    ///  In COO format, the index matrix of X is the following 4x6 matrix:
                    ///  ```text
                    ///    [[0, 0, 0, 0, 1, 1],
                    ///     [1, 1, 1, 2, 1, 2],
                    ///     [2, 2, 3, 1, 2, 0],
                    ///     [0, 1, 0, 0, 3, 4]]
                    ///  ```
                    ///  When isCanonical is true, the indices is sorted in lexicographical order
                    ///  (row-major order), and it does not have duplicated entries.  Otherwise,
                    ///  the indices may not be sorted, or may have duplicated entries.
                    ///
                    /// Generated from these locations:
                    /// * Table `SparseTensorIndexCOO` in the file `SparseTensor.fbs:58`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct SparseTensorIndexCoo {
                        ///  The type of values in indicesBuffer
                        pub indices_type: ::planus::alloc::boxed::Box<self::Int>,
                        ///  Non-negative byte offsets to advance one value cell along each dimension
                        ///  If omitted, default to row-major order (C-like).
                        pub indices_strides: ::core::option::Option<::planus::alloc::vec::Vec<i64>>,
                        ///  The location and size of the indices matrix's data
                        pub indices_buffer: self::Buffer,
                        ///  This flag is true if and only if the indices matrix is sorted in
                        ///  row-major order, and does not have duplicated entries.
                        ///  This sort order is the same as of Tensorflow's SparseTensor,
                        ///  but it is inverse order of SciPy's canonical coo_matrix
                        ///  (SciPy employs column-major order for its coo_matrix).
                        pub is_canonical: bool,
                    }

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for SparseTensorIndexCoo {
                        fn default() -> Self {
                            Self {
                                indices_type: ::core::default::Default::default(),
                                indices_strides: ::core::default::Default::default(),
                                indices_buffer: ::core::default::Default::default(),
                                is_canonical: false,
                            }
                        }
                    }

                    impl SparseTensorIndexCoo {
                        /// Creates a [SparseTensorIndexCooBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> SparseTensorIndexCooBuilder<()> {
                            SparseTensorIndexCooBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            field_indices_type: impl ::planus::WriteAs<::planus::Offset<self::Int>>,
                            field_indices_strides: impl ::planus::WriteAsOptional<
                                ::planus::Offset<[i64]>,
                            >,
                            field_indices_buffer: impl ::planus::WriteAs<self::Buffer>,
                            field_is_canonical: impl ::planus::WriteAsDefault<bool, bool>,
                        ) -> ::planus::Offset<Self> {
                            let prepared_indices_type = field_indices_type.prepare(builder);
                            let prepared_indices_strides = field_indices_strides.prepare(builder);
                            let prepared_indices_buffer = field_indices_buffer.prepare(builder);
                            let prepared_is_canonical = field_is_canonical.prepare(builder, &false);

                            let mut table_writer: ::planus::table_writer::TableWriter<12> =
                                ::core::default::Default::default();
                            table_writer.write_entry::<self::Buffer>(2);
                            table_writer.write_entry::<::planus::Offset<self::Int>>(0);
                            if prepared_indices_strides.is_some() {
                                table_writer.write_entry::<::planus::Offset<[i64]>>(1);
                            }
                            if prepared_is_canonical.is_some() {
                                table_writer.write_entry::<bool>(3);
                            }

                            unsafe {
                                table_writer.finish(builder, |object_writer| {
                                    object_writer.write::<_, _, 16>(&prepared_indices_buffer);
                                    object_writer.write::<_, _, 4>(&prepared_indices_type);
                                    if let ::core::option::Option::Some(prepared_indices_strides) =
                                        prepared_indices_strides
                                    {
                                        object_writer.write::<_, _, 4>(&prepared_indices_strides);
                                    }
                                    if let ::core::option::Option::Some(prepared_is_canonical) =
                                        prepared_is_canonical
                                    {
                                        object_writer.write::<_, _, 1>(&prepared_is_canonical);
                                    }
                                });
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<SparseTensorIndexCoo>> for SparseTensorIndexCoo {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<SparseTensorIndexCoo> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<SparseTensorIndexCoo>> for SparseTensorIndexCoo {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<SparseTensorIndexCoo>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<SparseTensorIndexCoo> for SparseTensorIndexCoo {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<SparseTensorIndexCoo> {
                            SparseTensorIndexCoo::create(
                                builder,
                                &self.indices_type,
                                &self.indices_strides,
                                self.indices_buffer,
                                self.is_canonical,
                            )
                        }
                    }

                    /// Builder for serializing an instance of the [SparseTensorIndexCoo] type.
                    ///
                    /// Can be created using the [SparseTensorIndexCoo::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct SparseTensorIndexCooBuilder<State>(State);

                    impl SparseTensorIndexCooBuilder<()> {
                        /// Setter for the [`indicesType` field](SparseTensorIndexCoo#structfield.indices_type).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn indices_type<T0>(
                            self,
                            value: T0,
                        ) -> SparseTensorIndexCooBuilder<(T0,)>
                        where
                            T0: ::planus::WriteAs<::planus::Offset<self::Int>>,
                        {
                            SparseTensorIndexCooBuilder((value,))
                        }
                    }

                    impl<T0> SparseTensorIndexCooBuilder<(T0,)> {
                        /// Setter for the [`indicesStrides` field](SparseTensorIndexCoo#structfield.indices_strides).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn indices_strides<T1>(
                            self,
                            value: T1,
                        ) -> SparseTensorIndexCooBuilder<(T0, T1)>
                        where
                            T1: ::planus::WriteAsOptional<::planus::Offset<[i64]>>,
                        {
                            let (v0,) = self.0;
                            SparseTensorIndexCooBuilder((v0, value))
                        }

                        /// Sets the [`indicesStrides` field](SparseTensorIndexCoo#structfield.indices_strides) to null.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn indices_strides_as_null(
                            self,
                        ) -> SparseTensorIndexCooBuilder<(T0, ())> {
                            self.indices_strides(())
                        }
                    }

                    impl<T0, T1> SparseTensorIndexCooBuilder<(T0, T1)> {
                        /// Setter for the [`indicesBuffer` field](SparseTensorIndexCoo#structfield.indices_buffer).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn indices_buffer<T2>(
                            self,
                            value: T2,
                        ) -> SparseTensorIndexCooBuilder<(T0, T1, T2)>
                        where
                            T2: ::planus::WriteAs<self::Buffer>,
                        {
                            let (v0, v1) = self.0;
                            SparseTensorIndexCooBuilder((v0, v1, value))
                        }
                    }

                    impl<T0, T1, T2> SparseTensorIndexCooBuilder<(T0, T1, T2)> {
                        /// Setter for the [`isCanonical` field](SparseTensorIndexCoo#structfield.is_canonical).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn is_canonical<T3>(
                            self,
                            value: T3,
                        ) -> SparseTensorIndexCooBuilder<(T0, T1, T2, T3)>
                        where
                            T3: ::planus::WriteAsDefault<bool, bool>,
                        {
                            let (v0, v1, v2) = self.0;
                            SparseTensorIndexCooBuilder((v0, v1, v2, value))
                        }

                        /// Sets the [`isCanonical` field](SparseTensorIndexCoo#structfield.is_canonical) to the default value.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn is_canonical_as_default(
                            self,
                        ) -> SparseTensorIndexCooBuilder<(T0, T1, T2, ::planus::DefaultValue)>
                        {
                            self.is_canonical(::planus::DefaultValue)
                        }
                    }

                    impl<T0, T1, T2, T3> SparseTensorIndexCooBuilder<(T0, T1, T2, T3)> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [SparseTensorIndexCoo].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<SparseTensorIndexCoo>
                        where
                            Self: ::planus::WriteAsOffset<SparseTensorIndexCoo>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAs<::planus::Offset<self::Int>>,
                            T1: ::planus::WriteAsOptional<::planus::Offset<[i64]>>,
                            T2: ::planus::WriteAs<self::Buffer>,
                            T3: ::planus::WriteAsDefault<bool, bool>,
                        >
                        ::planus::WriteAs<::planus::Offset<SparseTensorIndexCoo>>
                        for SparseTensorIndexCooBuilder<(T0, T1, T2, T3)>
                    {
                        type Prepared = ::planus::Offset<SparseTensorIndexCoo>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<SparseTensorIndexCoo> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAs<::planus::Offset<self::Int>>,
                            T1: ::planus::WriteAsOptional<::planus::Offset<[i64]>>,
                            T2: ::planus::WriteAs<self::Buffer>,
                            T3: ::planus::WriteAsDefault<bool, bool>,
                        >
                        ::planus::WriteAsOptional<::planus::Offset<SparseTensorIndexCoo>>
                        for SparseTensorIndexCooBuilder<(T0, T1, T2, T3)>
                    {
                        type Prepared = ::planus::Offset<SparseTensorIndexCoo>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<SparseTensorIndexCoo>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl<
                            T0: ::planus::WriteAs<::planus::Offset<self::Int>>,
                            T1: ::planus::WriteAsOptional<::planus::Offset<[i64]>>,
                            T2: ::planus::WriteAs<self::Buffer>,
                            T3: ::planus::WriteAsDefault<bool, bool>,
                        > ::planus::WriteAsOffset<SparseTensorIndexCoo>
                        for SparseTensorIndexCooBuilder<(T0, T1, T2, T3)>
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<SparseTensorIndexCoo> {
                            let (v0, v1, v2, v3) = &self.0;
                            SparseTensorIndexCoo::create(builder, v0, v1, v2, v3)
                        }
                    }

                    /// Reference to a deserialized [SparseTensorIndexCoo].
                    #[derive(Copy, Clone)]
                    pub struct SparseTensorIndexCooRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> SparseTensorIndexCooRef<'a> {
                        /// Getter for the [`indicesType` field](SparseTensorIndexCoo#structfield.indices_type).
                        #[inline]
                        pub fn indices_type(&self) -> ::planus::Result<self::IntRef<'a>> {
                            self.0
                                .access_required(0, "SparseTensorIndexCoo", "indices_type")
                        }

                        /// Getter for the [`indicesStrides` field](SparseTensorIndexCoo#structfield.indices_strides).
                        #[inline]
                        pub fn indices_strides(
                            &self,
                        ) -> ::planus::Result<::core::option::Option<::planus::Vector<'a, i64>>>
                        {
                            self.0.access(1, "SparseTensorIndexCoo", "indices_strides")
                        }

                        /// Getter for the [`indicesBuffer` field](SparseTensorIndexCoo#structfield.indices_buffer).
                        #[inline]
                        pub fn indices_buffer(&self) -> ::planus::Result<self::BufferRef<'a>> {
                            self.0
                                .access_required(2, "SparseTensorIndexCoo", "indices_buffer")
                        }

                        /// Getter for the [`isCanonical` field](SparseTensorIndexCoo#structfield.is_canonical).
                        #[inline]
                        pub fn is_canonical(&self) -> ::planus::Result<bool> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(3, "SparseTensorIndexCoo", "is_canonical")?
                                    .unwrap_or(false),
                            )
                        }
                    }

                    impl<'a> ::core::fmt::Debug for SparseTensorIndexCooRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("SparseTensorIndexCooRef");
                            f.field("indices_type", &self.indices_type());
                            if let ::core::option::Option::Some(field_indices_strides) =
                                self.indices_strides().transpose()
                            {
                                f.field("indices_strides", &field_indices_strides);
                            }
                            f.field("indices_buffer", &self.indices_buffer());
                            f.field("is_canonical", &self.is_canonical());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<SparseTensorIndexCooRef<'a>> for SparseTensorIndexCoo {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: SparseTensorIndexCooRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                indices_type: ::planus::alloc::boxed::Box::new(
                                    ::core::convert::TryInto::try_into(value.indices_type()?)?,
                                ),
                                indices_strides: if let ::core::option::Option::Some(
                                    indices_strides,
                                ) = value.indices_strides()?
                                {
                                    ::core::option::Option::Some(indices_strides.to_vec()?)
                                } else {
                                    ::core::option::Option::None
                                },
                                indices_buffer: ::core::convert::Into::into(
                                    value.indices_buffer()?,
                                ),
                                is_canonical: ::core::convert::TryInto::try_into(
                                    value.is_canonical()?,
                                )?,
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for SparseTensorIndexCooRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for SparseTensorIndexCooRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[SparseTensorIndexCooRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<SparseTensorIndexCoo>> for SparseTensorIndexCoo {
                        type Value = ::planus::Offset<SparseTensorIndexCoo>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<SparseTensorIndexCoo>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for SparseTensorIndexCooRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[SparseTensorIndexCooRef]",
                                    "read_as_root",
                                    0,
                                )
                            })
                        }
                    }

                    /// The enum `SparseMatrixCompressedAxis` in the namespace `org.apache.arrow.flatbuf`
                    ///
                    /// Generated from these locations:
                    /// * Enum `SparseMatrixCompressedAxis` in the file `SparseTensor.fbs:77`
                    #[derive(
                        Copy,
                        Clone,
                        Debug,
                        PartialEq,
                        Eq,
                        PartialOrd,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    #[repr(i16)]
                    pub enum SparseMatrixCompressedAxis {
                        /// The variant `Row` in the enum `SparseMatrixCompressedAxis`
                        Row = 0,

                        /// The variant `Column` in the enum `SparseMatrixCompressedAxis`
                        Column = 1,
                    }

                    impl SparseMatrixCompressedAxis {
                        /// Array containing all valid variants of SparseMatrixCompressedAxis
                        pub const ENUM_VALUES: [Self; 2] = [Self::Row, Self::Column];
                    }

                    impl ::core::convert::TryFrom<i16> for SparseMatrixCompressedAxis {
                        type Error = ::planus::errors::UnknownEnumTagKind;
                        #[inline]
                        fn try_from(
                            value: i16,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTagKind>
                        {
                            #[allow(clippy::match_single_binding)]
                            match value {
                                0 => ::core::result::Result::Ok(SparseMatrixCompressedAxis::Row),
                                1 => ::core::result::Result::Ok(SparseMatrixCompressedAxis::Column),

                                _ => ::core::result::Result::Err(
                                    ::planus::errors::UnknownEnumTagKind { tag: value as i128 },
                                ),
                            }
                        }
                    }

                    impl ::core::convert::From<SparseMatrixCompressedAxis> for i16 {
                        #[inline]
                        fn from(value: SparseMatrixCompressedAxis) -> Self {
                            value as i16
                        }
                    }

                    /// # Safety
                    /// The Planus compiler correctly calculates `ALIGNMENT` and `SIZE`.
                    unsafe impl ::planus::Primitive for SparseMatrixCompressedAxis {
                        const ALIGNMENT: usize = 2;
                        const SIZE: usize = 2;
                    }

                    impl ::planus::WriteAsPrimitive<SparseMatrixCompressedAxis> for SparseMatrixCompressedAxis {
                        #[inline]
                        fn write<const N: usize>(
                            &self,
                            cursor: ::planus::Cursor<'_, N>,
                            buffer_position: u32,
                        ) {
                            (*self as i16).write(cursor, buffer_position);
                        }
                    }

                    impl ::planus::WriteAs<SparseMatrixCompressedAxis> for SparseMatrixCompressedAxis {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                        ) -> SparseMatrixCompressedAxis {
                            *self
                        }
                    }

                    impl
                        ::planus::WriteAsDefault<
                            SparseMatrixCompressedAxis,
                            SparseMatrixCompressedAxis,
                        > for SparseMatrixCompressedAxis
                    {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                            default: &SparseMatrixCompressedAxis,
                        ) -> ::core::option::Option<SparseMatrixCompressedAxis>
                        {
                            if self == default {
                                ::core::option::Option::None
                            } else {
                                ::core::option::Option::Some(*self)
                            }
                        }
                    }

                    impl ::planus::WriteAsOptional<SparseMatrixCompressedAxis> for SparseMatrixCompressedAxis {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<SparseMatrixCompressedAxis>
                        {
                            ::core::option::Option::Some(*self)
                        }
                    }

                    impl<'buf> ::planus::TableRead<'buf> for SparseMatrixCompressedAxis {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            let n: i16 = ::planus::TableRead::from_buffer(buffer, offset)?;
                            ::core::result::Result::Ok(::core::convert::TryInto::try_into(n)?)
                        }
                    }

                    impl<'buf> ::planus::VectorReadInner<'buf> for SparseMatrixCompressedAxis {
                        type Error = ::planus::errors::UnknownEnumTag;
                        const STRIDE: usize = 2;
                        #[inline]
                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTag>
                        {
                            let value = unsafe {
                                <i16 as ::planus::VectorRead>::from_buffer(buffer, offset)
                            };
                            let value: ::core::result::Result<Self, _> =
                                ::core::convert::TryInto::try_into(value);
                            value.map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "SparseMatrixCompressedAxis",
                                    "VectorRead::from_buffer",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<SparseMatrixCompressedAxis> for SparseMatrixCompressedAxis {
                        const STRIDE: usize = 2;

                        type Value = Self;

                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self {
                            *self
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[Self],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 2];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (2 * i) as u32,
                                );
                            }
                        }
                    }

                    ///  Compressed Sparse format, that is matrix-specific.
                    ///
                    /// Generated from these locations:
                    /// * Table `SparseMatrixIndexCSX` in the file `SparseTensor.fbs:80`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct SparseMatrixIndexCsx {
                        ///  Which axis, row or column, is compressed
                        pub compressed_axis: self::SparseMatrixCompressedAxis,
                        ///  The type of values in indptrBuffer
                        pub indptr_type: ::planus::alloc::boxed::Box<self::Int>,
                        ///  indptrBuffer stores the location and size of indptr array that
                        ///  represents the range of the rows.
                        ///  The i-th row spans from `indptr[i]` to `indptr[i+1]` in the data.
                        ///  The length of this array is 1 + (the number of rows), and the type
                        ///  of index value is long.
                        ///
                        ///  For example, let X be the following 6x4 matrix:
                        ///  ```text
                        ///    X := [[0, 1, 2, 0],
                        ///          [0, 0, 3, 0],
                        ///          [0, 4, 0, 5],
                        ///          [0, 0, 0, 0],
                        ///          [6, 0, 7, 8],
                        ///          [0, 9, 0, 0]].
                        ///  ```
                        ///  The array of non-zero values in X is:
                        ///  ```text
                        ///    values(X) = [1, 2, 3, 4, 5, 6, 7, 8, 9].
                        ///  ```
                        ///  And the indptr of X is:
                        ///  ```text
                        ///    indptr(X) = [0, 2, 3, 5, 5, 8, 10].
                        ///  ```
                        pub indptr_buffer: self::Buffer,
                        ///  The type of values in indicesBuffer
                        pub indices_type: ::planus::alloc::boxed::Box<self::Int>,
                        ///  indicesBuffer stores the location and size of the array that
                        ///  contains the column indices of the corresponding non-zero values.
                        ///  The type of index value is long.
                        ///
                        ///  For example, the indices of the above X is:
                        ///  ```text
                        ///    indices(X) = [1, 2, 2, 1, 3, 0, 2, 3, 1].
                        ///  ```
                        ///  Note that the indices are sorted in lexicographical order for each row.
                        pub indices_buffer: self::Buffer,
                    }

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for SparseMatrixIndexCsx {
                        fn default() -> Self {
                            Self {
                                compressed_axis: self::SparseMatrixCompressedAxis::Row,
                                indptr_type: ::core::default::Default::default(),
                                indptr_buffer: ::core::default::Default::default(),
                                indices_type: ::core::default::Default::default(),
                                indices_buffer: ::core::default::Default::default(),
                            }
                        }
                    }

                    impl SparseMatrixIndexCsx {
                        /// Creates a [SparseMatrixIndexCsxBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> SparseMatrixIndexCsxBuilder<()> {
                            SparseMatrixIndexCsxBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            field_compressed_axis: impl ::planus::WriteAsDefault<
                                self::SparseMatrixCompressedAxis,
                                self::SparseMatrixCompressedAxis,
                            >,
                            field_indptr_type: impl ::planus::WriteAs<::planus::Offset<self::Int>>,
                            field_indptr_buffer: impl ::planus::WriteAs<self::Buffer>,
                            field_indices_type: impl ::planus::WriteAs<::planus::Offset<self::Int>>,
                            field_indices_buffer: impl ::planus::WriteAs<self::Buffer>,
                        ) -> ::planus::Offset<Self> {
                            let prepared_compressed_axis = field_compressed_axis
                                .prepare(builder, &self::SparseMatrixCompressedAxis::Row);
                            let prepared_indptr_type = field_indptr_type.prepare(builder);
                            let prepared_indptr_buffer = field_indptr_buffer.prepare(builder);
                            let prepared_indices_type = field_indices_type.prepare(builder);
                            let prepared_indices_buffer = field_indices_buffer.prepare(builder);

                            let mut table_writer: ::planus::table_writer::TableWriter<14> =
                                ::core::default::Default::default();
                            table_writer.write_entry::<self::Buffer>(2);
                            table_writer.write_entry::<self::Buffer>(4);
                            table_writer.write_entry::<::planus::Offset<self::Int>>(1);
                            table_writer.write_entry::<::planus::Offset<self::Int>>(3);
                            if prepared_compressed_axis.is_some() {
                                table_writer.write_entry::<self::SparseMatrixCompressedAxis>(0);
                            }

                            unsafe {
                                table_writer.finish(builder, |object_writer| {
                                    object_writer.write::<_, _, 16>(&prepared_indptr_buffer);
                                    object_writer.write::<_, _, 16>(&prepared_indices_buffer);
                                    object_writer.write::<_, _, 4>(&prepared_indptr_type);
                                    object_writer.write::<_, _, 4>(&prepared_indices_type);
                                    if let ::core::option::Option::Some(prepared_compressed_axis) =
                                        prepared_compressed_axis
                                    {
                                        object_writer.write::<_, _, 2>(&prepared_compressed_axis);
                                    }
                                });
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<SparseMatrixIndexCsx>> for SparseMatrixIndexCsx {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<SparseMatrixIndexCsx> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<SparseMatrixIndexCsx>> for SparseMatrixIndexCsx {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<SparseMatrixIndexCsx>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<SparseMatrixIndexCsx> for SparseMatrixIndexCsx {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<SparseMatrixIndexCsx> {
                            SparseMatrixIndexCsx::create(
                                builder,
                                self.compressed_axis,
                                &self.indptr_type,
                                self.indptr_buffer,
                                &self.indices_type,
                                self.indices_buffer,
                            )
                        }
                    }

                    /// Builder for serializing an instance of the [SparseMatrixIndexCsx] type.
                    ///
                    /// Can be created using the [SparseMatrixIndexCsx::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct SparseMatrixIndexCsxBuilder<State>(State);

                    impl SparseMatrixIndexCsxBuilder<()> {
                        /// Setter for the [`compressedAxis` field](SparseMatrixIndexCsx#structfield.compressed_axis).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn compressed_axis<T0>(
                            self,
                            value: T0,
                        ) -> SparseMatrixIndexCsxBuilder<(T0,)>
                        where
                            T0: ::planus::WriteAsDefault<
                                self::SparseMatrixCompressedAxis,
                                self::SparseMatrixCompressedAxis,
                            >,
                        {
                            SparseMatrixIndexCsxBuilder((value,))
                        }

                        /// Sets the [`compressedAxis` field](SparseMatrixIndexCsx#structfield.compressed_axis) to the default value.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn compressed_axis_as_default(
                            self,
                        ) -> SparseMatrixIndexCsxBuilder<(::planus::DefaultValue,)>
                        {
                            self.compressed_axis(::planus::DefaultValue)
                        }
                    }

                    impl<T0> SparseMatrixIndexCsxBuilder<(T0,)> {
                        /// Setter for the [`indptrType` field](SparseMatrixIndexCsx#structfield.indptr_type).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn indptr_type<T1>(
                            self,
                            value: T1,
                        ) -> SparseMatrixIndexCsxBuilder<(T0, T1)>
                        where
                            T1: ::planus::WriteAs<::planus::Offset<self::Int>>,
                        {
                            let (v0,) = self.0;
                            SparseMatrixIndexCsxBuilder((v0, value))
                        }
                    }

                    impl<T0, T1> SparseMatrixIndexCsxBuilder<(T0, T1)> {
                        /// Setter for the [`indptrBuffer` field](SparseMatrixIndexCsx#structfield.indptr_buffer).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn indptr_buffer<T2>(
                            self,
                            value: T2,
                        ) -> SparseMatrixIndexCsxBuilder<(T0, T1, T2)>
                        where
                            T2: ::planus::WriteAs<self::Buffer>,
                        {
                            let (v0, v1) = self.0;
                            SparseMatrixIndexCsxBuilder((v0, v1, value))
                        }
                    }

                    impl<T0, T1, T2> SparseMatrixIndexCsxBuilder<(T0, T1, T2)> {
                        /// Setter for the [`indicesType` field](SparseMatrixIndexCsx#structfield.indices_type).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn indices_type<T3>(
                            self,
                            value: T3,
                        ) -> SparseMatrixIndexCsxBuilder<(T0, T1, T2, T3)>
                        where
                            T3: ::planus::WriteAs<::planus::Offset<self::Int>>,
                        {
                            let (v0, v1, v2) = self.0;
                            SparseMatrixIndexCsxBuilder((v0, v1, v2, value))
                        }
                    }

                    impl<T0, T1, T2, T3> SparseMatrixIndexCsxBuilder<(T0, T1, T2, T3)> {
                        /// Setter for the [`indicesBuffer` field](SparseMatrixIndexCsx#structfield.indices_buffer).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn indices_buffer<T4>(
                            self,
                            value: T4,
                        ) -> SparseMatrixIndexCsxBuilder<(T0, T1, T2, T3, T4)>
                        where
                            T4: ::planus::WriteAs<self::Buffer>,
                        {
                            let (v0, v1, v2, v3) = self.0;
                            SparseMatrixIndexCsxBuilder((v0, v1, v2, v3, value))
                        }
                    }

                    impl<T0, T1, T2, T3, T4> SparseMatrixIndexCsxBuilder<(T0, T1, T2, T3, T4)> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [SparseMatrixIndexCsx].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<SparseMatrixIndexCsx>
                        where
                            Self: ::planus::WriteAsOffset<SparseMatrixIndexCsx>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<
                                self::SparseMatrixCompressedAxis,
                                self::SparseMatrixCompressedAxis,
                            >,
                            T1: ::planus::WriteAs<::planus::Offset<self::Int>>,
                            T2: ::planus::WriteAs<self::Buffer>,
                            T3: ::planus::WriteAs<::planus::Offset<self::Int>>,
                            T4: ::planus::WriteAs<self::Buffer>,
                        >
                        ::planus::WriteAs<::planus::Offset<SparseMatrixIndexCsx>>
                        for SparseMatrixIndexCsxBuilder<(T0, T1, T2, T3, T4)>
                    {
                        type Prepared = ::planus::Offset<SparseMatrixIndexCsx>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<SparseMatrixIndexCsx> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<
                                self::SparseMatrixCompressedAxis,
                                self::SparseMatrixCompressedAxis,
                            >,
                            T1: ::planus::WriteAs<::planus::Offset<self::Int>>,
                            T2: ::planus::WriteAs<self::Buffer>,
                            T3: ::planus::WriteAs<::planus::Offset<self::Int>>,
                            T4: ::planus::WriteAs<self::Buffer>,
                        >
                        ::planus::WriteAsOptional<::planus::Offset<SparseMatrixIndexCsx>>
                        for SparseMatrixIndexCsxBuilder<(T0, T1, T2, T3, T4)>
                    {
                        type Prepared = ::planus::Offset<SparseMatrixIndexCsx>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<SparseMatrixIndexCsx>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<
                                self::SparseMatrixCompressedAxis,
                                self::SparseMatrixCompressedAxis,
                            >,
                            T1: ::planus::WriteAs<::planus::Offset<self::Int>>,
                            T2: ::planus::WriteAs<self::Buffer>,
                            T3: ::planus::WriteAs<::planus::Offset<self::Int>>,
                            T4: ::planus::WriteAs<self::Buffer>,
                        > ::planus::WriteAsOffset<SparseMatrixIndexCsx>
                        for SparseMatrixIndexCsxBuilder<(T0, T1, T2, T3, T4)>
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<SparseMatrixIndexCsx> {
                            let (v0, v1, v2, v3, v4) = &self.0;
                            SparseMatrixIndexCsx::create(builder, v0, v1, v2, v3, v4)
                        }
                    }

                    /// Reference to a deserialized [SparseMatrixIndexCsx].
                    #[derive(Copy, Clone)]
                    pub struct SparseMatrixIndexCsxRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> SparseMatrixIndexCsxRef<'a> {
                        /// Getter for the [`compressedAxis` field](SparseMatrixIndexCsx#structfield.compressed_axis).
                        #[inline]
                        pub fn compressed_axis(
                            &self,
                        ) -> ::planus::Result<self::SparseMatrixCompressedAxis>
                        {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(0, "SparseMatrixIndexCsx", "compressed_axis")?
                                    .unwrap_or(self::SparseMatrixCompressedAxis::Row),
                            )
                        }

                        /// Getter for the [`indptrType` field](SparseMatrixIndexCsx#structfield.indptr_type).
                        #[inline]
                        pub fn indptr_type(&self) -> ::planus::Result<self::IntRef<'a>> {
                            self.0
                                .access_required(1, "SparseMatrixIndexCsx", "indptr_type")
                        }

                        /// Getter for the [`indptrBuffer` field](SparseMatrixIndexCsx#structfield.indptr_buffer).
                        #[inline]
                        pub fn indptr_buffer(&self) -> ::planus::Result<self::BufferRef<'a>> {
                            self.0
                                .access_required(2, "SparseMatrixIndexCsx", "indptr_buffer")
                        }

                        /// Getter for the [`indicesType` field](SparseMatrixIndexCsx#structfield.indices_type).
                        #[inline]
                        pub fn indices_type(&self) -> ::planus::Result<self::IntRef<'a>> {
                            self.0
                                .access_required(3, "SparseMatrixIndexCsx", "indices_type")
                        }

                        /// Getter for the [`indicesBuffer` field](SparseMatrixIndexCsx#structfield.indices_buffer).
                        #[inline]
                        pub fn indices_buffer(&self) -> ::planus::Result<self::BufferRef<'a>> {
                            self.0
                                .access_required(4, "SparseMatrixIndexCsx", "indices_buffer")
                        }
                    }

                    impl<'a> ::core::fmt::Debug for SparseMatrixIndexCsxRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("SparseMatrixIndexCsxRef");
                            f.field("compressed_axis", &self.compressed_axis());
                            f.field("indptr_type", &self.indptr_type());
                            f.field("indptr_buffer", &self.indptr_buffer());
                            f.field("indices_type", &self.indices_type());
                            f.field("indices_buffer", &self.indices_buffer());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<SparseMatrixIndexCsxRef<'a>> for SparseMatrixIndexCsx {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: SparseMatrixIndexCsxRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                compressed_axis: ::core::convert::TryInto::try_into(
                                    value.compressed_axis()?,
                                )?,
                                indptr_type: ::planus::alloc::boxed::Box::new(
                                    ::core::convert::TryInto::try_into(value.indptr_type()?)?,
                                ),
                                indptr_buffer: ::core::convert::Into::into(value.indptr_buffer()?),
                                indices_type: ::planus::alloc::boxed::Box::new(
                                    ::core::convert::TryInto::try_into(value.indices_type()?)?,
                                ),
                                indices_buffer: ::core::convert::Into::into(
                                    value.indices_buffer()?,
                                ),
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for SparseMatrixIndexCsxRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for SparseMatrixIndexCsxRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[SparseMatrixIndexCsxRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<SparseMatrixIndexCsx>> for SparseMatrixIndexCsx {
                        type Value = ::planus::Offset<SparseMatrixIndexCsx>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<SparseMatrixIndexCsx>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for SparseMatrixIndexCsxRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[SparseMatrixIndexCsxRef]",
                                    "read_as_root",
                                    0,
                                )
                            })
                        }
                    }

                    ///  Compressed Sparse Fiber (CSF) sparse tensor index.
                    ///
                    /// Generated from these locations:
                    /// * Table `SparseTensorIndexCSF` in the file `SparseTensor.fbs:128`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct SparseTensorIndexCsf {
                        ///  CSF is a generalization of compressed sparse row (CSR) index.
                        ///  See [smith2017knl](http://shaden.io/pub-files/smith2017knl.pdf)
                        ///
                        ///  CSF index recursively compresses each dimension of a tensor into a set
                        ///  of prefix trees. Each path from a root to leaf forms one tensor
                        ///  non-zero index. CSF is implemented with two arrays of buffers and one
                        ///  arrays of integers.
                        ///
                        ///  For example, let X be a 2x3x4x5 tensor and let it have the following
                        ///  8 non-zero values:
                        ///  ```text
                        ///    X[0, 0, 0, 1] := 1
                        ///    X[0, 0, 0, 2] := 2
                        ///    X[0, 1, 0, 0] := 3
                        ///    X[0, 1, 0, 2] := 4
                        ///    X[0, 1, 1, 0] := 5
                        ///    X[1, 1, 1, 0] := 6
                        ///    X[1, 1, 1, 1] := 7
                        ///    X[1, 1, 1, 2] := 8
                        ///  ```
                        ///  As a prefix tree this would be represented as:
                        ///  ```text
                        ///          0          1
                        ///         / \         |
                        ///        0   1        1
                        ///       /   / \       |
                        ///      0   0   1      1
                        ///     /|  /|   |    /| |
                        ///    1 2 0 2   0   0 1 2
                        ///  ```
                        ///  The type of values in indptrBuffers
                        pub indptr_type: ::planus::alloc::boxed::Box<self::Int>,
                        ///  indptrBuffers stores the sparsity structure.
                        ///  Each two consecutive dimensions in a tensor correspond to a buffer in
                        ///  indptrBuffers. A pair of consecutive values at `indptrBuffers[dim][i]`
                        ///  and `indptrBuffers[dim][i + 1]` signify a range of nodes in
                        ///  `indicesBuffers[dim + 1]` who are children of `indicesBuffers[dim][i]` node.
                        ///
                        ///  For example, the indptrBuffers for the above X is:
                        ///  ```text
                        ///    indptrBuffer(X) = [
                        ///                        [0, 2, 3],
                        ///                        [0, 1, 3, 4],
                        ///                        [0, 2, 4, 5, 8]
                        ///                      ].
                        ///  ```
                        pub indptr_buffers: ::planus::alloc::vec::Vec<self::Buffer>,
                        ///  The type of values in indicesBuffers
                        pub indices_type: ::planus::alloc::boxed::Box<self::Int>,
                        ///  indicesBuffers stores values of nodes.
                        ///  Each tensor dimension corresponds to a buffer in indicesBuffers.
                        ///  For example, the indicesBuffers for the above X is:
                        ///  ```text
                        ///    indicesBuffer(X) = [
                        ///                         [0, 1],
                        ///                         [0, 1, 1],
                        ///                         [0, 0, 1, 1],
                        ///                         [1, 2, 0, 2, 0, 0, 1, 2]
                        ///                       ].
                        ///  ```
                        pub indices_buffers: ::planus::alloc::vec::Vec<self::Buffer>,
                        ///  axisOrder stores the sequence in which dimensions were traversed to
                        ///  produce the prefix tree.
                        ///  For example, the axisOrder for the above X is:
                        ///  ```text
                        ///    axisOrder(X) = [0, 1, 2, 3].
                        ///  ```
                        pub axis_order: ::planus::alloc::vec::Vec<i32>,
                    }

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for SparseTensorIndexCsf {
                        fn default() -> Self {
                            Self {
                                indptr_type: ::core::default::Default::default(),
                                indptr_buffers: ::core::default::Default::default(),
                                indices_type: ::core::default::Default::default(),
                                indices_buffers: ::core::default::Default::default(),
                                axis_order: ::core::default::Default::default(),
                            }
                        }
                    }

                    impl SparseTensorIndexCsf {
                        /// Creates a [SparseTensorIndexCsfBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> SparseTensorIndexCsfBuilder<()> {
                            SparseTensorIndexCsfBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            field_indptr_type: impl ::planus::WriteAs<::planus::Offset<self::Int>>,
                            field_indptr_buffers: impl ::planus::WriteAs<
                                ::planus::Offset<[self::Buffer]>,
                            >,
                            field_indices_type: impl ::planus::WriteAs<::planus::Offset<self::Int>>,
                            field_indices_buffers: impl ::planus::WriteAs<
                                ::planus::Offset<[self::Buffer]>,
                            >,
                            field_axis_order: impl ::planus::WriteAs<::planus::Offset<[i32]>>,
                        ) -> ::planus::Offset<Self> {
                            let prepared_indptr_type = field_indptr_type.prepare(builder);
                            let prepared_indptr_buffers = field_indptr_buffers.prepare(builder);
                            let prepared_indices_type = field_indices_type.prepare(builder);
                            let prepared_indices_buffers = field_indices_buffers.prepare(builder);
                            let prepared_axis_order = field_axis_order.prepare(builder);

                            let mut table_writer: ::planus::table_writer::TableWriter<14> =
                                ::core::default::Default::default();
                            table_writer.write_entry::<::planus::Offset<self::Int>>(0);
                            table_writer.write_entry::<::planus::Offset<[self::Buffer]>>(1);
                            table_writer.write_entry::<::planus::Offset<self::Int>>(2);
                            table_writer.write_entry::<::planus::Offset<[self::Buffer]>>(3);
                            table_writer.write_entry::<::planus::Offset<[i32]>>(4);

                            unsafe {
                                table_writer.finish(builder, |object_writer| {
                                    object_writer.write::<_, _, 4>(&prepared_indptr_type);
                                    object_writer.write::<_, _, 4>(&prepared_indptr_buffers);
                                    object_writer.write::<_, _, 4>(&prepared_indices_type);
                                    object_writer.write::<_, _, 4>(&prepared_indices_buffers);
                                    object_writer.write::<_, _, 4>(&prepared_axis_order);
                                });
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<SparseTensorIndexCsf>> for SparseTensorIndexCsf {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<SparseTensorIndexCsf> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<SparseTensorIndexCsf>> for SparseTensorIndexCsf {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<SparseTensorIndexCsf>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<SparseTensorIndexCsf> for SparseTensorIndexCsf {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<SparseTensorIndexCsf> {
                            SparseTensorIndexCsf::create(
                                builder,
                                &self.indptr_type,
                                &self.indptr_buffers,
                                &self.indices_type,
                                &self.indices_buffers,
                                &self.axis_order,
                            )
                        }
                    }

                    /// Builder for serializing an instance of the [SparseTensorIndexCsf] type.
                    ///
                    /// Can be created using the [SparseTensorIndexCsf::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct SparseTensorIndexCsfBuilder<State>(State);

                    impl SparseTensorIndexCsfBuilder<()> {
                        /// Setter for the [`indptrType` field](SparseTensorIndexCsf#structfield.indptr_type).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn indptr_type<T0>(
                            self,
                            value: T0,
                        ) -> SparseTensorIndexCsfBuilder<(T0,)>
                        where
                            T0: ::planus::WriteAs<::planus::Offset<self::Int>>,
                        {
                            SparseTensorIndexCsfBuilder((value,))
                        }
                    }

                    impl<T0> SparseTensorIndexCsfBuilder<(T0,)> {
                        /// Setter for the [`indptrBuffers` field](SparseTensorIndexCsf#structfield.indptr_buffers).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn indptr_buffers<T1>(
                            self,
                            value: T1,
                        ) -> SparseTensorIndexCsfBuilder<(T0, T1)>
                        where
                            T1: ::planus::WriteAs<::planus::Offset<[self::Buffer]>>,
                        {
                            let (v0,) = self.0;
                            SparseTensorIndexCsfBuilder((v0, value))
                        }
                    }

                    impl<T0, T1> SparseTensorIndexCsfBuilder<(T0, T1)> {
                        /// Setter for the [`indicesType` field](SparseTensorIndexCsf#structfield.indices_type).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn indices_type<T2>(
                            self,
                            value: T2,
                        ) -> SparseTensorIndexCsfBuilder<(T0, T1, T2)>
                        where
                            T2: ::planus::WriteAs<::planus::Offset<self::Int>>,
                        {
                            let (v0, v1) = self.0;
                            SparseTensorIndexCsfBuilder((v0, v1, value))
                        }
                    }

                    impl<T0, T1, T2> SparseTensorIndexCsfBuilder<(T0, T1, T2)> {
                        /// Setter for the [`indicesBuffers` field](SparseTensorIndexCsf#structfield.indices_buffers).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn indices_buffers<T3>(
                            self,
                            value: T3,
                        ) -> SparseTensorIndexCsfBuilder<(T0, T1, T2, T3)>
                        where
                            T3: ::planus::WriteAs<::planus::Offset<[self::Buffer]>>,
                        {
                            let (v0, v1, v2) = self.0;
                            SparseTensorIndexCsfBuilder((v0, v1, v2, value))
                        }
                    }

                    impl<T0, T1, T2, T3> SparseTensorIndexCsfBuilder<(T0, T1, T2, T3)> {
                        /// Setter for the [`axisOrder` field](SparseTensorIndexCsf#structfield.axis_order).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn axis_order<T4>(
                            self,
                            value: T4,
                        ) -> SparseTensorIndexCsfBuilder<(T0, T1, T2, T3, T4)>
                        where
                            T4: ::planus::WriteAs<::planus::Offset<[i32]>>,
                        {
                            let (v0, v1, v2, v3) = self.0;
                            SparseTensorIndexCsfBuilder((v0, v1, v2, v3, value))
                        }
                    }

                    impl<T0, T1, T2, T3, T4> SparseTensorIndexCsfBuilder<(T0, T1, T2, T3, T4)> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [SparseTensorIndexCsf].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<SparseTensorIndexCsf>
                        where
                            Self: ::planus::WriteAsOffset<SparseTensorIndexCsf>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAs<::planus::Offset<self::Int>>,
                            T1: ::planus::WriteAs<::planus::Offset<[self::Buffer]>>,
                            T2: ::planus::WriteAs<::planus::Offset<self::Int>>,
                            T3: ::planus::WriteAs<::planus::Offset<[self::Buffer]>>,
                            T4: ::planus::WriteAs<::planus::Offset<[i32]>>,
                        >
                        ::planus::WriteAs<::planus::Offset<SparseTensorIndexCsf>>
                        for SparseTensorIndexCsfBuilder<(T0, T1, T2, T3, T4)>
                    {
                        type Prepared = ::planus::Offset<SparseTensorIndexCsf>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<SparseTensorIndexCsf> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAs<::planus::Offset<self::Int>>,
                            T1: ::planus::WriteAs<::planus::Offset<[self::Buffer]>>,
                            T2: ::planus::WriteAs<::planus::Offset<self::Int>>,
                            T3: ::planus::WriteAs<::planus::Offset<[self::Buffer]>>,
                            T4: ::planus::WriteAs<::planus::Offset<[i32]>>,
                        >
                        ::planus::WriteAsOptional<::planus::Offset<SparseTensorIndexCsf>>
                        for SparseTensorIndexCsfBuilder<(T0, T1, T2, T3, T4)>
                    {
                        type Prepared = ::planus::Offset<SparseTensorIndexCsf>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<SparseTensorIndexCsf>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl<
                            T0: ::planus::WriteAs<::planus::Offset<self::Int>>,
                            T1: ::planus::WriteAs<::planus::Offset<[self::Buffer]>>,
                            T2: ::planus::WriteAs<::planus::Offset<self::Int>>,
                            T3: ::planus::WriteAs<::planus::Offset<[self::Buffer]>>,
                            T4: ::planus::WriteAs<::planus::Offset<[i32]>>,
                        > ::planus::WriteAsOffset<SparseTensorIndexCsf>
                        for SparseTensorIndexCsfBuilder<(T0, T1, T2, T3, T4)>
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<SparseTensorIndexCsf> {
                            let (v0, v1, v2, v3, v4) = &self.0;
                            SparseTensorIndexCsf::create(builder, v0, v1, v2, v3, v4)
                        }
                    }

                    /// Reference to a deserialized [SparseTensorIndexCsf].
                    #[derive(Copy, Clone)]
                    pub struct SparseTensorIndexCsfRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> SparseTensorIndexCsfRef<'a> {
                        /// Getter for the [`indptrType` field](SparseTensorIndexCsf#structfield.indptr_type).
                        #[inline]
                        pub fn indptr_type(&self) -> ::planus::Result<self::IntRef<'a>> {
                            self.0
                                .access_required(0, "SparseTensorIndexCsf", "indptr_type")
                        }

                        /// Getter for the [`indptrBuffers` field](SparseTensorIndexCsf#structfield.indptr_buffers).
                        #[inline]
                        pub fn indptr_buffers(
                            &self,
                        ) -> ::planus::Result<::planus::Vector<'a, self::BufferRef<'a>>>
                        {
                            self.0
                                .access_required(1, "SparseTensorIndexCsf", "indptr_buffers")
                        }

                        /// Getter for the [`indicesType` field](SparseTensorIndexCsf#structfield.indices_type).
                        #[inline]
                        pub fn indices_type(&self) -> ::planus::Result<self::IntRef<'a>> {
                            self.0
                                .access_required(2, "SparseTensorIndexCsf", "indices_type")
                        }

                        /// Getter for the [`indicesBuffers` field](SparseTensorIndexCsf#structfield.indices_buffers).
                        #[inline]
                        pub fn indices_buffers(
                            &self,
                        ) -> ::planus::Result<::planus::Vector<'a, self::BufferRef<'a>>>
                        {
                            self.0
                                .access_required(3, "SparseTensorIndexCsf", "indices_buffers")
                        }

                        /// Getter for the [`axisOrder` field](SparseTensorIndexCsf#structfield.axis_order).
                        #[inline]
                        pub fn axis_order(&self) -> ::planus::Result<::planus::Vector<'a, i32>> {
                            self.0
                                .access_required(4, "SparseTensorIndexCsf", "axis_order")
                        }
                    }

                    impl<'a> ::core::fmt::Debug for SparseTensorIndexCsfRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("SparseTensorIndexCsfRef");
                            f.field("indptr_type", &self.indptr_type());
                            f.field("indptr_buffers", &self.indptr_buffers());
                            f.field("indices_type", &self.indices_type());
                            f.field("indices_buffers", &self.indices_buffers());
                            f.field("axis_order", &self.axis_order());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<SparseTensorIndexCsfRef<'a>> for SparseTensorIndexCsf {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: SparseTensorIndexCsfRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                indptr_type: ::planus::alloc::boxed::Box::new(
                                    ::core::convert::TryInto::try_into(value.indptr_type()?)?,
                                ),
                                indptr_buffers: value.indptr_buffers()?.to_vec()?,
                                indices_type: ::planus::alloc::boxed::Box::new(
                                    ::core::convert::TryInto::try_into(value.indices_type()?)?,
                                ),
                                indices_buffers: value.indices_buffers()?.to_vec()?,
                                axis_order: value.axis_order()?.to_vec()?,
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for SparseTensorIndexCsfRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for SparseTensorIndexCsfRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[SparseTensorIndexCsfRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<SparseTensorIndexCsf>> for SparseTensorIndexCsf {
                        type Value = ::planus::Offset<SparseTensorIndexCsf>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<SparseTensorIndexCsf>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for SparseTensorIndexCsfRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[SparseTensorIndexCsfRef]",
                                    "read_as_root",
                                    0,
                                )
                            })
                        }
                    }

                    /// The union `SparseTensorIndex` in the namespace `org.apache.arrow.flatbuf`
                    ///
                    /// Generated from these locations:
                    /// * Union `SparseTensorIndex` in the file `SparseTensor.fbs:203`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub enum SparseTensorIndex {
                        /// The variant of type `SparseTensorIndexCOO` in the union `SparseTensorIndex`
                        SparseTensorIndexCoo(
                            ::planus::alloc::boxed::Box<self::SparseTensorIndexCoo>,
                        ),

                        /// The variant of type `SparseMatrixIndexCSX` in the union `SparseTensorIndex`
                        SparseMatrixIndexCsx(
                            ::planus::alloc::boxed::Box<self::SparseMatrixIndexCsx>,
                        ),

                        /// The variant of type `SparseTensorIndexCSF` in the union `SparseTensorIndex`
                        SparseTensorIndexCsf(
                            ::planus::alloc::boxed::Box<self::SparseTensorIndexCsf>,
                        ),
                    }

                    impl SparseTensorIndex {
                        /// Creates a [SparseTensorIndexBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> SparseTensorIndexBuilder<::planus::Uninitialized> {
                            SparseTensorIndexBuilder(::planus::Uninitialized)
                        }

                        #[inline]
                        pub fn create_sparse_tensor_index_coo(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::SparseTensorIndexCoo>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(1, value.prepare(builder).downcast())
                        }

                        #[inline]
                        pub fn create_sparse_matrix_index_csx(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::SparseMatrixIndexCsx>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(2, value.prepare(builder).downcast())
                        }

                        #[inline]
                        pub fn create_sparse_tensor_index_csf(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::SparseTensorIndexCsf>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(3, value.prepare(builder).downcast())
                        }
                    }

                    impl ::planus::WriteAsUnion<SparseTensorIndex> for SparseTensorIndex {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<Self> {
                            match self {
                                Self::SparseTensorIndexCoo(value) => {
                                    Self::create_sparse_tensor_index_coo(builder, value)
                                }
                                Self::SparseMatrixIndexCsx(value) => {
                                    Self::create_sparse_matrix_index_csx(builder, value)
                                }
                                Self::SparseTensorIndexCsf(value) => {
                                    Self::create_sparse_tensor_index_csf(builder, value)
                                }
                            }
                        }
                    }

                    impl ::planus::WriteAsOptionalUnion<SparseTensorIndex> for SparseTensorIndex {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<Self>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }

                    /// Builder for serializing an instance of the [SparseTensorIndex] type.
                    ///
                    /// Can be created using the [SparseTensorIndex::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct SparseTensorIndexBuilder<T>(T);

                    impl SparseTensorIndexBuilder<::planus::Uninitialized> {
                        /// Creates an instance of the [`SparseTensorIndexCOO` variant](SparseTensorIndex#variant.SparseTensorIndexCoo).
                        #[inline]
                        pub fn sparse_tensor_index_coo<T>(
                            self,
                            value: T,
                        ) -> SparseTensorIndexBuilder<::planus::Initialized<1, T>>
                        where
                            T: ::planus::WriteAsOffset<self::SparseTensorIndexCoo>,
                        {
                            SparseTensorIndexBuilder(::planus::Initialized(value))
                        }

                        /// Creates an instance of the [`SparseMatrixIndexCSX` variant](SparseTensorIndex#variant.SparseMatrixIndexCsx).
                        #[inline]
                        pub fn sparse_matrix_index_csx<T>(
                            self,
                            value: T,
                        ) -> SparseTensorIndexBuilder<::planus::Initialized<2, T>>
                        where
                            T: ::planus::WriteAsOffset<self::SparseMatrixIndexCsx>,
                        {
                            SparseTensorIndexBuilder(::planus::Initialized(value))
                        }

                        /// Creates an instance of the [`SparseTensorIndexCSF` variant](SparseTensorIndex#variant.SparseTensorIndexCsf).
                        #[inline]
                        pub fn sparse_tensor_index_csf<T>(
                            self,
                            value: T,
                        ) -> SparseTensorIndexBuilder<::planus::Initialized<3, T>>
                        where
                            T: ::planus::WriteAsOffset<self::SparseTensorIndexCsf>,
                        {
                            SparseTensorIndexBuilder(::planus::Initialized(value))
                        }
                    }

                    impl<const N: u8, T> SparseTensorIndexBuilder<::planus::Initialized<N, T>> {
                        /// Finish writing the builder to get an [UnionOffset](::planus::UnionOffset) to a serialized [SparseTensorIndex].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<SparseTensorIndex>
                        where
                            Self: ::planus::WriteAsUnion<SparseTensorIndex>,
                        {
                            ::planus::WriteAsUnion::prepare(&self, builder)
                        }
                    }

                    impl<T> ::planus::WriteAsUnion<SparseTensorIndex>
                        for SparseTensorIndexBuilder<::planus::Initialized<1, T>>
                    where
                        T: ::planus::WriteAsOffset<self::SparseTensorIndexCoo>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<SparseTensorIndex> {
                            ::planus::UnionOffset::new(1, (self.0).0.prepare(builder).downcast())
                        }
                    }

                    impl<T> ::planus::WriteAsOptionalUnion<SparseTensorIndex>
                        for SparseTensorIndexBuilder<::planus::Initialized<1, T>>
                    where
                        T: ::planus::WriteAsOffset<self::SparseTensorIndexCoo>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<SparseTensorIndex>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }
                    impl<T> ::planus::WriteAsUnion<SparseTensorIndex>
                        for SparseTensorIndexBuilder<::planus::Initialized<2, T>>
                    where
                        T: ::planus::WriteAsOffset<self::SparseMatrixIndexCsx>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<SparseTensorIndex> {
                            ::planus::UnionOffset::new(2, (self.0).0.prepare(builder).downcast())
                        }
                    }

                    impl<T> ::planus::WriteAsOptionalUnion<SparseTensorIndex>
                        for SparseTensorIndexBuilder<::planus::Initialized<2, T>>
                    where
                        T: ::planus::WriteAsOffset<self::SparseMatrixIndexCsx>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<SparseTensorIndex>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }
                    impl<T> ::planus::WriteAsUnion<SparseTensorIndex>
                        for SparseTensorIndexBuilder<::planus::Initialized<3, T>>
                    where
                        T: ::planus::WriteAsOffset<self::SparseTensorIndexCsf>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<SparseTensorIndex> {
                            ::planus::UnionOffset::new(3, (self.0).0.prepare(builder).downcast())
                        }
                    }

                    impl<T> ::planus::WriteAsOptionalUnion<SparseTensorIndex>
                        for SparseTensorIndexBuilder<::planus::Initialized<3, T>>
                    where
                        T: ::planus::WriteAsOffset<self::SparseTensorIndexCsf>,
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<SparseTensorIndex>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }

                    /// Reference to a deserialized [SparseTensorIndex].
                    #[derive(Copy, Clone, Debug)]
                    pub enum SparseTensorIndexRef<'a> {
                        SparseTensorIndexCoo(self::SparseTensorIndexCooRef<'a>),
                        SparseMatrixIndexCsx(self::SparseMatrixIndexCsxRef<'a>),
                        SparseTensorIndexCsf(self::SparseTensorIndexCsfRef<'a>),
                    }

                    impl<'a> ::core::convert::TryFrom<SparseTensorIndexRef<'a>> for SparseTensorIndex {
                        type Error = ::planus::Error;

                        fn try_from(value: SparseTensorIndexRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(match value {
                                SparseTensorIndexRef::SparseTensorIndexCoo(value) => {
                                    Self::SparseTensorIndexCoo(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                SparseTensorIndexRef::SparseMatrixIndexCsx(value) => {
                                    Self::SparseMatrixIndexCsx(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                SparseTensorIndexRef::SparseTensorIndexCsf(value) => {
                                    Self::SparseTensorIndexCsf(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }
                            })
                        }
                    }

                    impl<'a> ::planus::TableReadUnion<'a> for SparseTensorIndexRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            tag: u8,
                            field_offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            match tag {
                                1 => ::core::result::Result::Ok(Self::SparseTensorIndexCoo(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                2 => ::core::result::Result::Ok(Self::SparseMatrixIndexCsx(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                3 => ::core::result::Result::Ok(Self::SparseTensorIndexCsf(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                _ => ::core::result::Result::Err(
                                    ::planus::errors::ErrorKind::UnknownUnionTag { tag },
                                ),
                            }
                        }
                    }

                    impl<'a> ::planus::VectorReadUnion<'a> for SparseTensorIndexRef<'a> {
                        const VECTOR_NAME: &'static str = "[SparseTensorIndexRef]";
                    }

                    /// The table `SparseTensor` in the namespace `org.apache.arrow.flatbuf`
                    ///
                    /// Generated from these locations:
                    /// * Table `SparseTensor` in the file `SparseTensor.fbs:209`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct SparseTensor {
                        ///  The type of data contained in a value cell.
                        ///  Currently only fixed-width value types are supported,
                        ///  no strings or nested types.
                        pub type_: self::Type,
                        ///  The dimensions of the tensor, optionally named.
                        pub shape: ::planus::alloc::vec::Vec<self::TensorDim>,
                        ///  The number of non-zero values in a sparse tensor.
                        pub non_zero_length: i64,
                        ///  Sparse tensor index
                        pub sparse_index: self::SparseTensorIndex,
                        ///  The location and size of the tensor's data
                        pub data: self::Buffer,
                    }

                    impl SparseTensor {
                        /// Creates a [SparseTensorBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> SparseTensorBuilder<()> {
                            SparseTensorBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            field_type_: impl ::planus::WriteAsUnion<self::Type>,
                            field_shape: impl ::planus::WriteAs<
                                ::planus::Offset<[::planus::Offset<self::TensorDim>]>,
                            >,
                            field_non_zero_length: impl ::planus::WriteAsDefault<i64, i64>,
                            field_sparse_index: impl ::planus::WriteAsUnion<self::SparseTensorIndex>,
                            field_data: impl ::planus::WriteAs<self::Buffer>,
                        ) -> ::planus::Offset<Self> {
                            let prepared_type_ = field_type_.prepare(builder);
                            let prepared_shape = field_shape.prepare(builder);
                            let prepared_non_zero_length =
                                field_non_zero_length.prepare(builder, &0);
                            let prepared_sparse_index = field_sparse_index.prepare(builder);
                            let prepared_data = field_data.prepare(builder);

                            let mut table_writer: ::planus::table_writer::TableWriter<18> =
                                ::core::default::Default::default();
                            if prepared_non_zero_length.is_some() {
                                table_writer.write_entry::<i64>(3);
                            }
                            table_writer.write_entry::<self::Buffer>(6);
                            table_writer.write_entry::<::planus::Offset<self::Type>>(1);
                            table_writer.write_entry::<::planus::Offset<[::planus::Offset<self::TensorDim>]>>(2);
                            table_writer
                                .write_entry::<::planus::Offset<self::SparseTensorIndex>>(5);
                            table_writer.write_entry::<u8>(0);
                            table_writer.write_entry::<u8>(4);

                            unsafe {
                                table_writer.finish(builder, |object_writer| {
                                    if let ::core::option::Option::Some(prepared_non_zero_length) =
                                        prepared_non_zero_length
                                    {
                                        object_writer.write::<_, _, 8>(&prepared_non_zero_length);
                                    }
                                    object_writer.write::<_, _, 16>(&prepared_data);
                                    object_writer.write::<_, _, 4>(&prepared_type_.offset());
                                    object_writer.write::<_, _, 4>(&prepared_shape);
                                    object_writer.write::<_, _, 4>(&prepared_sparse_index.offset());
                                    object_writer.write::<_, _, 1>(&prepared_type_.tag());
                                    object_writer.write::<_, _, 1>(&prepared_sparse_index.tag());
                                });
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<SparseTensor>> for SparseTensor {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<SparseTensor> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<SparseTensor>> for SparseTensor {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<SparseTensor>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<SparseTensor> for SparseTensor {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<SparseTensor> {
                            SparseTensor::create(
                                builder,
                                &self.type_,
                                &self.shape,
                                self.non_zero_length,
                                &self.sparse_index,
                                self.data,
                            )
                        }
                    }

                    /// Builder for serializing an instance of the [SparseTensor] type.
                    ///
                    /// Can be created using the [SparseTensor::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct SparseTensorBuilder<State>(State);

                    impl SparseTensorBuilder<()> {
                        /// Setter for the [`type` field](SparseTensor#structfield.type_).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn type_<T0>(self, value: T0) -> SparseTensorBuilder<(T0,)>
                        where
                            T0: ::planus::WriteAsUnion<self::Type>,
                        {
                            SparseTensorBuilder((value,))
                        }
                    }

                    impl<T0> SparseTensorBuilder<(T0,)> {
                        /// Setter for the [`shape` field](SparseTensor#structfield.shape).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn shape<T1>(self, value: T1) -> SparseTensorBuilder<(T0, T1)>
                        where
                            T1: ::planus::WriteAs<
                                ::planus::Offset<[::planus::Offset<self::TensorDim>]>,
                            >,
                        {
                            let (v0,) = self.0;
                            SparseTensorBuilder((v0, value))
                        }
                    }

                    impl<T0, T1> SparseTensorBuilder<(T0, T1)> {
                        /// Setter for the [`non_zero_length` field](SparseTensor#structfield.non_zero_length).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn non_zero_length<T2>(
                            self,
                            value: T2,
                        ) -> SparseTensorBuilder<(T0, T1, T2)>
                        where
                            T2: ::planus::WriteAsDefault<i64, i64>,
                        {
                            let (v0, v1) = self.0;
                            SparseTensorBuilder((v0, v1, value))
                        }

                        /// Sets the [`non_zero_length` field](SparseTensor#structfield.non_zero_length) to the default value.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn non_zero_length_as_default(
                            self,
                        ) -> SparseTensorBuilder<(T0, T1, ::planus::DefaultValue)>
                        {
                            self.non_zero_length(::planus::DefaultValue)
                        }
                    }

                    impl<T0, T1, T2> SparseTensorBuilder<(T0, T1, T2)> {
                        /// Setter for the [`sparseIndex` field](SparseTensor#structfield.sparse_index).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn sparse_index<T3>(
                            self,
                            value: T3,
                        ) -> SparseTensorBuilder<(T0, T1, T2, T3)>
                        where
                            T3: ::planus::WriteAsUnion<self::SparseTensorIndex>,
                        {
                            let (v0, v1, v2) = self.0;
                            SparseTensorBuilder((v0, v1, v2, value))
                        }
                    }

                    impl<T0, T1, T2, T3> SparseTensorBuilder<(T0, T1, T2, T3)> {
                        /// Setter for the [`data` field](SparseTensor#structfield.data).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn data<T4>(
                            self,
                            value: T4,
                        ) -> SparseTensorBuilder<(T0, T1, T2, T3, T4)>
                        where
                            T4: ::planus::WriteAs<self::Buffer>,
                        {
                            let (v0, v1, v2, v3) = self.0;
                            SparseTensorBuilder((v0, v1, v2, v3, value))
                        }
                    }

                    impl<T0, T1, T2, T3, T4> SparseTensorBuilder<(T0, T1, T2, T3, T4)> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [SparseTensor].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<SparseTensor>
                        where
                            Self: ::planus::WriteAsOffset<SparseTensor>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsUnion<self::Type>,
                            T1: ::planus::WriteAs<
                                ::planus::Offset<[::planus::Offset<self::TensorDim>]>,
                            >,
                            T2: ::planus::WriteAsDefault<i64, i64>,
                            T3: ::planus::WriteAsUnion<self::SparseTensorIndex>,
                            T4: ::planus::WriteAs<self::Buffer>,
                        > ::planus::WriteAs<::planus::Offset<SparseTensor>>
                        for SparseTensorBuilder<(T0, T1, T2, T3, T4)>
                    {
                        type Prepared = ::planus::Offset<SparseTensor>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<SparseTensor> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsUnion<self::Type>,
                            T1: ::planus::WriteAs<
                                ::planus::Offset<[::planus::Offset<self::TensorDim>]>,
                            >,
                            T2: ::planus::WriteAsDefault<i64, i64>,
                            T3: ::planus::WriteAsUnion<self::SparseTensorIndex>,
                            T4: ::planus::WriteAs<self::Buffer>,
                        >
                        ::planus::WriteAsOptional<::planus::Offset<SparseTensor>>
                        for SparseTensorBuilder<(T0, T1, T2, T3, T4)>
                    {
                        type Prepared = ::planus::Offset<SparseTensor>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<SparseTensor>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsUnion<self::Type>,
                            T1: ::planus::WriteAs<
                                ::planus::Offset<[::planus::Offset<self::TensorDim>]>,
                            >,
                            T2: ::planus::WriteAsDefault<i64, i64>,
                            T3: ::planus::WriteAsUnion<self::SparseTensorIndex>,
                            T4: ::planus::WriteAs<self::Buffer>,
                        > ::planus::WriteAsOffset<SparseTensor>
                        for SparseTensorBuilder<(T0, T1, T2, T3, T4)>
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<SparseTensor> {
                            let (v0, v1, v2, v3, v4) = &self.0;
                            SparseTensor::create(builder, v0, v1, v2, v3, v4)
                        }
                    }

                    /// Reference to a deserialized [SparseTensor].
                    #[derive(Copy, Clone)]
                    pub struct SparseTensorRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> SparseTensorRef<'a> {
                        /// Getter for the [`type` field](SparseTensor#structfield.type_).
                        #[inline]
                        pub fn type_(&self) -> ::planus::Result<self::TypeRef<'a>> {
                            self.0.access_union_required(0, "SparseTensor", "type_")
                        }

                        /// Getter for the [`shape` field](SparseTensor#structfield.shape).
                        #[inline]
                        pub fn shape(
                            &self,
                        ) -> ::planus::Result<
                            ::planus::Vector<'a, ::planus::Result<self::TensorDimRef<'a>>>,
                        > {
                            self.0.access_required(2, "SparseTensor", "shape")
                        }

                        /// Getter for the [`non_zero_length` field](SparseTensor#structfield.non_zero_length).
                        #[inline]
                        pub fn non_zero_length(&self) -> ::planus::Result<i64> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(3, "SparseTensor", "non_zero_length")?
                                    .unwrap_or(0),
                            )
                        }

                        /// Getter for the [`sparseIndex` field](SparseTensor#structfield.sparse_index).
                        #[inline]
                        pub fn sparse_index(
                            &self,
                        ) -> ::planus::Result<self::SparseTensorIndexRef<'a>>
                        {
                            self.0
                                .access_union_required(4, "SparseTensor", "sparse_index")
                        }

                        /// Getter for the [`data` field](SparseTensor#structfield.data).
                        #[inline]
                        pub fn data(&self) -> ::planus::Result<self::BufferRef<'a>> {
                            self.0.access_required(6, "SparseTensor", "data")
                        }
                    }

                    impl<'a> ::core::fmt::Debug for SparseTensorRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("SparseTensorRef");
                            f.field("type_", &self.type_());
                            f.field("shape", &self.shape());
                            f.field("non_zero_length", &self.non_zero_length());
                            f.field("sparse_index", &self.sparse_index());
                            f.field("data", &self.data());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<SparseTensorRef<'a>> for SparseTensor {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: SparseTensorRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                type_: ::core::convert::TryInto::try_into(value.type_()?)?,
                                shape: value.shape()?.to_vec_result()?,
                                non_zero_length: ::core::convert::TryInto::try_into(
                                    value.non_zero_length()?,
                                )?,
                                sparse_index: ::core::convert::TryInto::try_into(
                                    value.sparse_index()?,
                                )?,
                                data: ::core::convert::Into::into(value.data()?),
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for SparseTensorRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for SparseTensorRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[SparseTensorRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<SparseTensor>> for SparseTensor {
                        type Value = ::planus::Offset<SparseTensor>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<SparseTensor>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for SparseTensorRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[SparseTensorRef]",
                                    "read_as_root",
                                    0,
                                )
                            })
                        }
                    }

                    ///  ----------------------------------------------------------------------
                    ///  Data structures for dense tensors
                    ///  Shape data for a single axis in a tensor
                    ///
                    /// Generated from these locations:
                    /// * Table `TensorDim` in the file `Tensor.fbs:30`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct TensorDim {
                        ///  Length of dimension
                        pub size: i64,
                        ///  Name of the dimension, optional
                        pub name: ::core::option::Option<::planus::alloc::string::String>,
                    }

                    #[allow(clippy::derivable_impls)]
                    impl ::core::default::Default for TensorDim {
                        fn default() -> Self {
                            Self {
                                size: 0,
                                name: ::core::default::Default::default(),
                            }
                        }
                    }

                    impl TensorDim {
                        /// Creates a [TensorDimBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> TensorDimBuilder<()> {
                            TensorDimBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            field_size: impl ::planus::WriteAsDefault<i64, i64>,
                            field_name: impl ::planus::WriteAsOptional<
                                ::planus::Offset<::core::primitive::str>,
                            >,
                        ) -> ::planus::Offset<Self> {
                            let prepared_size = field_size.prepare(builder, &0);
                            let prepared_name = field_name.prepare(builder);

                            let mut table_writer: ::planus::table_writer::TableWriter<8> =
                                ::core::default::Default::default();
                            if prepared_size.is_some() {
                                table_writer.write_entry::<i64>(0);
                            }
                            if prepared_name.is_some() {
                                table_writer.write_entry::<::planus::Offset<str>>(1);
                            }

                            unsafe {
                                table_writer.finish(builder, |object_writer| {
                                    if let ::core::option::Option::Some(prepared_size) =
                                        prepared_size
                                    {
                                        object_writer.write::<_, _, 8>(&prepared_size);
                                    }
                                    if let ::core::option::Option::Some(prepared_name) =
                                        prepared_name
                                    {
                                        object_writer.write::<_, _, 4>(&prepared_name);
                                    }
                                });
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<TensorDim>> for TensorDim {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<TensorDim> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<TensorDim>> for TensorDim {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<TensorDim>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<TensorDim> for TensorDim {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<TensorDim> {
                            TensorDim::create(builder, self.size, &self.name)
                        }
                    }

                    /// Builder for serializing an instance of the [TensorDim] type.
                    ///
                    /// Can be created using the [TensorDim::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct TensorDimBuilder<State>(State);

                    impl TensorDimBuilder<()> {
                        /// Setter for the [`size` field](TensorDim#structfield.size).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn size<T0>(self, value: T0) -> TensorDimBuilder<(T0,)>
                        where
                            T0: ::planus::WriteAsDefault<i64, i64>,
                        {
                            TensorDimBuilder((value,))
                        }

                        /// Sets the [`size` field](TensorDim#structfield.size) to the default value.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn size_as_default(
                            self,
                        ) -> TensorDimBuilder<(::planus::DefaultValue,)> {
                            self.size(::planus::DefaultValue)
                        }
                    }

                    impl<T0> TensorDimBuilder<(T0,)> {
                        /// Setter for the [`name` field](TensorDim#structfield.name).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn name<T1>(self, value: T1) -> TensorDimBuilder<(T0, T1)>
                        where
                            T1: ::planus::WriteAsOptional<::planus::Offset<::core::primitive::str>>,
                        {
                            let (v0,) = self.0;
                            TensorDimBuilder((v0, value))
                        }

                        /// Sets the [`name` field](TensorDim#structfield.name) to null.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn name_as_null(self) -> TensorDimBuilder<(T0, ())> {
                            self.name(())
                        }
                    }

                    impl<T0, T1> TensorDimBuilder<(T0, T1)> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [TensorDim].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<TensorDim>
                        where
                            Self: ::planus::WriteAsOffset<TensorDim>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<i64, i64>,
                            T1: ::planus::WriteAsOptional<::planus::Offset<::core::primitive::str>>,
                        > ::planus::WriteAs<::planus::Offset<TensorDim>>
                        for TensorDimBuilder<(T0, T1)>
                    {
                        type Prepared = ::planus::Offset<TensorDim>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<TensorDim> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<i64, i64>,
                            T1: ::planus::WriteAsOptional<::planus::Offset<::core::primitive::str>>,
                        > ::planus::WriteAsOptional<::planus::Offset<TensorDim>>
                        for TensorDimBuilder<(T0, T1)>
                    {
                        type Prepared = ::planus::Offset<TensorDim>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<TensorDim>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsDefault<i64, i64>,
                            T1: ::planus::WriteAsOptional<::planus::Offset<::core::primitive::str>>,
                        > ::planus::WriteAsOffset<TensorDim> for TensorDimBuilder<(T0, T1)>
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<TensorDim> {
                            let (v0, v1) = &self.0;
                            TensorDim::create(builder, v0, v1)
                        }
                    }

                    /// Reference to a deserialized [TensorDim].
                    #[derive(Copy, Clone)]
                    pub struct TensorDimRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> TensorDimRef<'a> {
                        /// Getter for the [`size` field](TensorDim#structfield.size).
                        #[inline]
                        pub fn size(&self) -> ::planus::Result<i64> {
                            ::core::result::Result::Ok(
                                self.0.access(0, "TensorDim", "size")?.unwrap_or(0),
                            )
                        }

                        /// Getter for the [`name` field](TensorDim#structfield.name).
                        #[inline]
                        pub fn name(
                            &self,
                        ) -> ::planus::Result<::core::option::Option<&'a ::core::primitive::str>>
                        {
                            self.0.access(1, "TensorDim", "name")
                        }
                    }

                    impl<'a> ::core::fmt::Debug for TensorDimRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("TensorDimRef");
                            f.field("size", &self.size());
                            if let ::core::option::Option::Some(field_name) =
                                self.name().transpose()
                            {
                                f.field("name", &field_name);
                            }
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<TensorDimRef<'a>> for TensorDim {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: TensorDimRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                size: ::core::convert::TryInto::try_into(value.size()?)?,
                                name: value.name()?.map(::core::convert::Into::into),
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for TensorDimRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for TensorDimRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[TensorDimRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<TensorDim>> for TensorDim {
                        type Value = ::planus::Offset<TensorDim>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<TensorDim>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for TensorDimRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[TensorDimRef]", "read_as_root", 0)
                            })
                        }
                    }

                    /// The table `Tensor` in the namespace `org.apache.arrow.flatbuf`
                    ///
                    /// Generated from these locations:
                    /// * Table `Tensor` in the file `Tensor.fbs:38`
                    #[derive(
                        Clone,
                        Debug,
                        PartialEq,
                        PartialOrd,
                        Eq,
                        Ord,
                        Hash,
                        ::serde::Serialize,
                        ::serde::Deserialize,
                    )]
                    pub struct Tensor {
                        ///  The type of data contained in a value cell. Currently only fixed-width
                        ///  value types are supported, no strings or nested types
                        pub type_: self::Type,
                        ///  The dimensions of the tensor, optionally named
                        pub shape: ::planus::alloc::vec::Vec<self::TensorDim>,
                        ///  Non-negative byte offsets to advance one value cell along each dimension
                        ///  If omitted, default to row-major order (C-like).
                        pub strides: ::core::option::Option<::planus::alloc::vec::Vec<i64>>,
                        ///  The location and size of the tensor's data
                        pub data: self::Buffer,
                    }

                    impl Tensor {
                        /// Creates a [TensorBuilder] for serializing an instance of this table.
                        #[inline]
                        pub fn builder() -> TensorBuilder<()> {
                            TensorBuilder(())
                        }

                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            field_type_: impl ::planus::WriteAsUnion<self::Type>,
                            field_shape: impl ::planus::WriteAs<
                                ::planus::Offset<[::planus::Offset<self::TensorDim>]>,
                            >,
                            field_strides: impl ::planus::WriteAsOptional<::planus::Offset<[i64]>>,
                            field_data: impl ::planus::WriteAs<self::Buffer>,
                        ) -> ::planus::Offset<Self> {
                            let prepared_type_ = field_type_.prepare(builder);
                            let prepared_shape = field_shape.prepare(builder);
                            let prepared_strides = field_strides.prepare(builder);
                            let prepared_data = field_data.prepare(builder);

                            let mut table_writer: ::planus::table_writer::TableWriter<14> =
                                ::core::default::Default::default();
                            table_writer.write_entry::<self::Buffer>(4);
                            table_writer.write_entry::<::planus::Offset<self::Type>>(1);
                            table_writer.write_entry::<::planus::Offset<[::planus::Offset<self::TensorDim>]>>(2);
                            if prepared_strides.is_some() {
                                table_writer.write_entry::<::planus::Offset<[i64]>>(3);
                            }
                            table_writer.write_entry::<u8>(0);

                            unsafe {
                                table_writer.finish(builder, |object_writer| {
                                    object_writer.write::<_, _, 16>(&prepared_data);
                                    object_writer.write::<_, _, 4>(&prepared_type_.offset());
                                    object_writer.write::<_, _, 4>(&prepared_shape);
                                    if let ::core::option::Option::Some(prepared_strides) =
                                        prepared_strides
                                    {
                                        object_writer.write::<_, _, 4>(&prepared_strides);
                                    }
                                    object_writer.write::<_, _, 1>(&prepared_type_.tag());
                                });
                            }
                            builder.current_offset()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Tensor>> for Tensor {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Tensor> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Tensor>> for Tensor {
                        type Prepared = ::planus::Offset<Self>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Tensor>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Tensor> for Tensor {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Tensor> {
                            Tensor::create(
                                builder,
                                &self.type_,
                                &self.shape,
                                &self.strides,
                                self.data,
                            )
                        }
                    }

                    /// Builder for serializing an instance of the [Tensor] type.
                    ///
                    /// Can be created using the [Tensor::builder] method.
                    #[derive(Debug)]
                    #[must_use]
                    pub struct TensorBuilder<State>(State);

                    impl TensorBuilder<()> {
                        /// Setter for the [`type` field](Tensor#structfield.type_).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn type_<T0>(self, value: T0) -> TensorBuilder<(T0,)>
                        where
                            T0: ::planus::WriteAsUnion<self::Type>,
                        {
                            TensorBuilder((value,))
                        }
                    }

                    impl<T0> TensorBuilder<(T0,)> {
                        /// Setter for the [`shape` field](Tensor#structfield.shape).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn shape<T1>(self, value: T1) -> TensorBuilder<(T0, T1)>
                        where
                            T1: ::planus::WriteAs<
                                ::planus::Offset<[::planus::Offset<self::TensorDim>]>,
                            >,
                        {
                            let (v0,) = self.0;
                            TensorBuilder((v0, value))
                        }
                    }

                    impl<T0, T1> TensorBuilder<(T0, T1)> {
                        /// Setter for the [`strides` field](Tensor#structfield.strides).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn strides<T2>(self, value: T2) -> TensorBuilder<(T0, T1, T2)>
                        where
                            T2: ::planus::WriteAsOptional<::planus::Offset<[i64]>>,
                        {
                            let (v0, v1) = self.0;
                            TensorBuilder((v0, v1, value))
                        }

                        /// Sets the [`strides` field](Tensor#structfield.strides) to null.
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn strides_as_null(self) -> TensorBuilder<(T0, T1, ())> {
                            self.strides(())
                        }
                    }

                    impl<T0, T1, T2> TensorBuilder<(T0, T1, T2)> {
                        /// Setter for the [`data` field](Tensor#structfield.data).
                        #[inline]
                        #[allow(clippy::type_complexity)]
                        pub fn data<T3>(self, value: T3) -> TensorBuilder<(T0, T1, T2, T3)>
                        where
                            T3: ::planus::WriteAs<self::Buffer>,
                        {
                            let (v0, v1, v2) = self.0;
                            TensorBuilder((v0, v1, v2, value))
                        }
                    }

                    impl<T0, T1, T2, T3> TensorBuilder<(T0, T1, T2, T3)> {
                        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Tensor].
                        #[inline]
                        pub fn finish(
                            self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Tensor>
                        where
                            Self: ::planus::WriteAsOffset<Tensor>,
                        {
                            ::planus::WriteAsOffset::prepare(&self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsUnion<self::Type>,
                            T1: ::planus::WriteAs<
                                ::planus::Offset<[::planus::Offset<self::TensorDim>]>,
                            >,
                            T2: ::planus::WriteAsOptional<::planus::Offset<[i64]>>,
                            T3: ::planus::WriteAs<self::Buffer>,
                        > ::planus::WriteAs<::planus::Offset<Tensor>>
                        for TensorBuilder<(T0, T1, T2, T3)>
                    {
                        type Prepared = ::planus::Offset<Tensor>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Tensor> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsUnion<self::Type>,
                            T1: ::planus::WriteAs<
                                ::planus::Offset<[::planus::Offset<self::TensorDim>]>,
                            >,
                            T2: ::planus::WriteAsOptional<::planus::Offset<[i64]>>,
                            T3: ::planus::WriteAs<self::Buffer>,
                        > ::planus::WriteAsOptional<::planus::Offset<Tensor>>
                        for TensorBuilder<(T0, T1, T2, T3)>
                    {
                        type Prepared = ::planus::Offset<Tensor>;

                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Tensor>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl<
                            T0: ::planus::WriteAsUnion<self::Type>,
                            T1: ::planus::WriteAs<
                                ::planus::Offset<[::planus::Offset<self::TensorDim>]>,
                            >,
                            T2: ::planus::WriteAsOptional<::planus::Offset<[i64]>>,
                            T3: ::planus::WriteAs<self::Buffer>,
                        > ::planus::WriteAsOffset<Tensor> for TensorBuilder<(T0, T1, T2, T3)>
                    {
                        #[inline]
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Tensor> {
                            let (v0, v1, v2, v3) = &self.0;
                            Tensor::create(builder, v0, v1, v2, v3)
                        }
                    }

                    /// Reference to a deserialized [Tensor].
                    #[derive(Copy, Clone)]
                    pub struct TensorRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> TensorRef<'a> {
                        /// Getter for the [`type` field](Tensor#structfield.type_).
                        #[inline]
                        pub fn type_(&self) -> ::planus::Result<self::TypeRef<'a>> {
                            self.0.access_union_required(0, "Tensor", "type_")
                        }

                        /// Getter for the [`shape` field](Tensor#structfield.shape).
                        #[inline]
                        pub fn shape(
                            &self,
                        ) -> ::planus::Result<
                            ::planus::Vector<'a, ::planus::Result<self::TensorDimRef<'a>>>,
                        > {
                            self.0.access_required(2, "Tensor", "shape")
                        }

                        /// Getter for the [`strides` field](Tensor#structfield.strides).
                        #[inline]
                        pub fn strides(
                            &self,
                        ) -> ::planus::Result<::core::option::Option<::planus::Vector<'a, i64>>>
                        {
                            self.0.access(3, "Tensor", "strides")
                        }

                        /// Getter for the [`data` field](Tensor#structfield.data).
                        #[inline]
                        pub fn data(&self) -> ::planus::Result<self::BufferRef<'a>> {
                            self.0.access_required(4, "Tensor", "data")
                        }
                    }

                    impl<'a> ::core::fmt::Debug for TensorRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("TensorRef");
                            f.field("type_", &self.type_());
                            f.field("shape", &self.shape());
                            if let ::core::option::Option::Some(field_strides) =
                                self.strides().transpose()
                            {
                                f.field("strides", &field_strides);
                            }
                            f.field("data", &self.data());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<TensorRef<'a>> for Tensor {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: TensorRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                type_: ::core::convert::TryInto::try_into(value.type_()?)?,
                                shape: value.shape()?.to_vec_result()?,
                                strides: if let ::core::option::Option::Some(strides) =
                                    value.strides()?
                                {
                                    ::core::option::Option::Some(strides.to_vec()?)
                                } else {
                                    ::core::option::Option::None
                                },
                                data: ::core::convert::Into::into(value.data()?),
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for TensorRef<'a> {
                        #[inline]
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for TensorRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[TensorRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    /// # Safety
                    /// The planus compiler generates implementations that initialize
                    /// the bytes in `write_values`.
                    unsafe impl ::planus::VectorWrite<::planus::Offset<Tensor>> for Tensor {
                        type Value = ::planus::Offset<Tensor>;
                        const STRIDE: usize = 4;
                        #[inline]
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Tensor>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for TensorRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[TensorRef]", "read_as_root", 0)
                            })
                        }
                    }
                }
            }
        }
    }
}
