use super::data::*;

/// Generated client implementations.
pub mod flight_service_client {
    #![allow(unused_variables, dead_code, missing_docs, clippy::let_unit_value)]
    use tonic::codegen::http::Uri;
    use tonic::codegen::*;
    ///
    /// A flight service is an endpoint for retrieving or storing Arrow data. A
    /// flight service can expose one or more predefined endpoints that can be
    /// accessed using the Arrow Flight Protocol. Additionally, a flight service
    /// can expose a set of actions that are available.
    #[derive(Debug, Clone)]
    pub struct FlightServiceClient<T> {
        inner: tonic::client::Grpc<T>,
    }
    impl FlightServiceClient<tonic::transport::Channel> {
        /// Attempt to create a new client by connecting to a given endpoint.
        pub async fn connect<D>(dst: D) -> Result<Self, tonic::transport::Error>
        where
            D: std::convert::TryInto<tonic::transport::Endpoint>,
            D::Error: Into<StdError>,
        {
            let conn = tonic::transport::Endpoint::new(dst)?.connect().await?;
            Ok(Self::new(conn))
        }
    }
    impl<T> FlightServiceClient<T>
    where
        T: tonic::client::GrpcService<tonic::body::BoxBody>,
        T::Error: Into<StdError>,
        T::ResponseBody: Body<Data = Bytes> + Send + 'static,
        <T::ResponseBody as Body>::Error: Into<StdError> + Send,
    {
        pub fn new(inner: T) -> Self {
            let inner = tonic::client::Grpc::new(inner);
            Self { inner }
        }
        pub fn with_origin(inner: T, origin: Uri) -> Self {
            let inner = tonic::client::Grpc::with_origin(inner, origin);
            Self { inner }
        }
        pub fn with_interceptor<F>(
            inner: T,
            interceptor: F,
        ) -> FlightServiceClient<InterceptedService<T, F>>
        where
            F: tonic::service::Interceptor,
            T::ResponseBody: Default,
            T: tonic::codegen::Service<
                http::Request<tonic::body::BoxBody>,
                Response = http::Response<
                    <T as tonic::client::GrpcService<tonic::body::BoxBody>>::ResponseBody,
                >,
            >,
            <T as tonic::codegen::Service<http::Request<tonic::body::BoxBody>>>::Error:
                Into<StdError> + Send + Sync,
        {
            FlightServiceClient::new(InterceptedService::new(inner, interceptor))
        }
        /// Compress requests with the given encoding.
        ///
        /// This requires the server to support it otherwise it might respond with an
        /// error.
        #[must_use]
        pub fn send_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.inner = self.inner.send_compressed(encoding);
            self
        }
        /// Enable decompressing responses.
        #[must_use]
        pub fn accept_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.inner = self.inner.accept_compressed(encoding);
            self
        }
        ///
        /// Handshake between client and server. Depending on the server, the
        /// handshake may be required to determine the token that should be used for
        /// future operations. Both request and response are streams to allow multiple
        /// round-trips depending on auth mechanism.
        pub async fn handshake(
            &mut self,
            request: impl tonic::IntoStreamingRequest<Message = super::HandshakeRequest>,
        ) -> Result<tonic::Response<tonic::codec::Streaming<super::HandshakeResponse>>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::new(
                    tonic::Code::Unknown,
                    format!("Service was not ready: {}", e.into()),
                )
            })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/arrow.flight.protocol.FlightService/Handshake",
            );
            self.inner
                .streaming(request.into_streaming_request(), path, codec)
                .await
        }
        ///
        /// Get a list of available streams given a particular criteria. Most flight
        /// services will expose one or more streams that are readily available for
        /// retrieval. This api allows listing the streams available for
        /// consumption. A user can also provide a criteria. The criteria can limit
        /// the subset of streams that can be listed via this interface. Each flight
        /// service allows its own definition of how to consume criteria.
        pub async fn list_flights(
            &mut self,
            request: impl tonic::IntoRequest<super::Criteria>,
        ) -> Result<tonic::Response<tonic::codec::Streaming<super::FlightInfo>>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::new(
                    tonic::Code::Unknown,
                    format!("Service was not ready: {}", e.into()),
                )
            })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/arrow.flight.protocol.FlightService/ListFlights",
            );
            self.inner
                .server_streaming(request.into_request(), path, codec)
                .await
        }
        ///
        /// For a given FlightDescriptor, get information about how the flight can be
        /// consumed. This is a useful interface if the consumer of the interface
        /// already can identify the specific flight to consume. This interface can
        /// also allow a consumer to generate a flight stream through a specified
        /// descriptor. For example, a flight descriptor might be something that
        /// includes a SQL statement or a Pickled Python operation that will be
        /// executed. In those cases, the descriptor will not be previously available
        /// within the list of available streams provided by ListFlights but will be
        /// available for consumption for the duration defined by the specific flight
        /// service.
        pub async fn get_flight_info(
            &mut self,
            request: impl tonic::IntoRequest<super::FlightDescriptor>,
        ) -> Result<tonic::Response<super::FlightInfo>, tonic::Status> {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::new(
                    tonic::Code::Unknown,
                    format!("Service was not ready: {}", e.into()),
                )
            })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/arrow.flight.protocol.FlightService/GetFlightInfo",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
        ///
        /// For a given FlightDescriptor, get the Schema as described in Schema.fbs::Schema
        /// This is used when a consumer needs the Schema of flight stream. Similar to
        /// GetFlightInfo this interface may generate a new flight that was not previously
        /// available in ListFlights.
        pub async fn get_schema(
            &mut self,
            request: impl tonic::IntoRequest<super::FlightDescriptor>,
        ) -> Result<tonic::Response<super::SchemaResult>, tonic::Status> {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::new(
                    tonic::Code::Unknown,
                    format!("Service was not ready: {}", e.into()),
                )
            })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/arrow.flight.protocol.FlightService/GetSchema",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
        ///
        /// Retrieve a single stream associated with a particular descriptor
        /// associated with the referenced ticket. A Flight can be composed of one or
        /// more streams where each stream can be retrieved using a separate opaque
        /// ticket that the flight service uses for managing a collection of streams.
        pub async fn do_get(
            &mut self,
            request: impl tonic::IntoRequest<super::Ticket>,
        ) -> Result<tonic::Response<tonic::codec::Streaming<super::FlightData>>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::new(
                    tonic::Code::Unknown,
                    format!("Service was not ready: {}", e.into()),
                )
            })?;
            let codec = tonic::codec::ProstCodec::default();
            let path =
                http::uri::PathAndQuery::from_static("/arrow.flight.protocol.FlightService/DoGet");
            self.inner
                .server_streaming(request.into_request(), path, codec)
                .await
        }
        ///
        /// Push a stream to the flight service associated with a particular
        /// flight stream. This allows a client of a flight service to upload a stream
        /// of data. Depending on the particular flight service, a client consumer
        /// could be allowed to upload a single stream per descriptor or an unlimited
        /// number. In the latter, the service might implement a 'seal' action that
        /// can be applied to a descriptor once all streams are uploaded.
        pub async fn do_put(
            &mut self,
            request: impl tonic::IntoStreamingRequest<Message = super::FlightData>,
        ) -> Result<tonic::Response<tonic::codec::Streaming<super::PutResult>>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::new(
                    tonic::Code::Unknown,
                    format!("Service was not ready: {}", e.into()),
                )
            })?;
            let codec = tonic::codec::ProstCodec::default();
            let path =
                http::uri::PathAndQuery::from_static("/arrow.flight.protocol.FlightService/DoPut");
            self.inner
                .streaming(request.into_streaming_request(), path, codec)
                .await
        }
        ///
        /// Open a bidirectional data channel for a given descriptor. This
        /// allows clients to send and receive arbitrary Arrow data and
        /// application-specific metadata in a single logical stream. In
        /// contrast to DoGet/DoPut, this is more suited for clients
        /// offloading computation (rather than storage) to a Flight service.
        pub async fn do_exchange(
            &mut self,
            request: impl tonic::IntoStreamingRequest<Message = super::FlightData>,
        ) -> Result<tonic::Response<tonic::codec::Streaming<super::FlightData>>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::new(
                    tonic::Code::Unknown,
                    format!("Service was not ready: {}", e.into()),
                )
            })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/arrow.flight.protocol.FlightService/DoExchange",
            );
            self.inner
                .streaming(request.into_streaming_request(), path, codec)
                .await
        }
        ///
        /// Flight services can support an arbitrary number of simple actions in
        /// addition to the possible ListFlights, GetFlightInfo, DoGet, DoPut
        /// operations that are potentially available. DoAction allows a flight client
        /// to do a specific action against a flight service. An action includes
        /// opaque request and response objects that are specific to the type action
        /// being undertaken.
        pub async fn do_action(
            &mut self,
            request: impl tonic::IntoRequest<super::Action>,
        ) -> Result<tonic::Response<tonic::codec::Streaming<super::Result>>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::new(
                    tonic::Code::Unknown,
                    format!("Service was not ready: {}", e.into()),
                )
            })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/arrow.flight.protocol.FlightService/DoAction",
            );
            self.inner
                .server_streaming(request.into_request(), path, codec)
                .await
        }
        ///
        /// A flight service exposes all of the available action types that it has
        /// along with descriptions. This allows different flight consumers to
        /// understand the capabilities of the flight service.
        pub async fn list_actions(
            &mut self,
            request: impl tonic::IntoRequest<super::Empty>,
        ) -> Result<tonic::Response<tonic::codec::Streaming<super::ActionType>>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::new(
                    tonic::Code::Unknown,
                    format!("Service was not ready: {}", e.into()),
                )
            })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/arrow.flight.protocol.FlightService/ListActions",
            );
            self.inner
                .server_streaming(request.into_request(), path, codec)
                .await
        }
    }
}
/// Generated server implementations.
pub mod flight_service_server {
    #![allow(unused_variables, dead_code, missing_docs, clippy::let_unit_value)]
    use tonic::codegen::*;
    ///Generated trait containing gRPC methods that should be implemented for use with FlightServiceServer.
    #[async_trait]
    pub trait FlightService: Send + Sync + 'static {
        ///Server streaming response type for the Handshake method.
        type HandshakeStream: futures_core::Stream<Item = Result<super::HandshakeResponse, tonic::Status>>
            + Send
            + 'static;
        ///
        /// Handshake between client and server. Depending on the server, the
        /// handshake may be required to determine the token that should be used for
        /// future operations. Both request and response are streams to allow multiple
        /// round-trips depending on auth mechanism.
        async fn handshake(
            &self,
            request: tonic::Request<tonic::Streaming<super::HandshakeRequest>>,
        ) -> Result<tonic::Response<Self::HandshakeStream>, tonic::Status>;
        ///Server streaming response type for the ListFlights method.
        type ListFlightsStream: futures_core::Stream<Item = Result<super::FlightInfo, tonic::Status>>
            + Send
            + 'static;
        ///
        /// Get a list of available streams given a particular criteria. Most flight
        /// services will expose one or more streams that are readily available for
        /// retrieval. This api allows listing the streams available for
        /// consumption. A user can also provide a criteria. The criteria can limit
        /// the subset of streams that can be listed via this interface. Each flight
        /// service allows its own definition of how to consume criteria.
        async fn list_flights(
            &self,
            request: tonic::Request<super::Criteria>,
        ) -> Result<tonic::Response<Self::ListFlightsStream>, tonic::Status>;
        ///
        /// For a given FlightDescriptor, get information about how the flight can be
        /// consumed. This is a useful interface if the consumer of the interface
        /// already can identify the specific flight to consume. This interface can
        /// also allow a consumer to generate a flight stream through a specified
        /// descriptor. For example, a flight descriptor might be something that
        /// includes a SQL statement or a Pickled Python operation that will be
        /// executed. In those cases, the descriptor will not be previously available
        /// within the list of available streams provided by ListFlights but will be
        /// available for consumption for the duration defined by the specific flight
        /// service.
        async fn get_flight_info(
            &self,
            request: tonic::Request<super::FlightDescriptor>,
        ) -> Result<tonic::Response<super::FlightInfo>, tonic::Status>;
        ///
        /// For a given FlightDescriptor, get the Schema as described in Schema.fbs::Schema
        /// This is used when a consumer needs the Schema of flight stream. Similar to
        /// GetFlightInfo this interface may generate a new flight that was not previously
        /// available in ListFlights.
        async fn get_schema(
            &self,
            request: tonic::Request<super::FlightDescriptor>,
        ) -> Result<tonic::Response<super::SchemaResult>, tonic::Status>;
        ///Server streaming response type for the DoGet method.
        type DoGetStream: futures_core::Stream<Item = Result<super::FlightData, tonic::Status>>
            + Send
            + 'static;
        ///
        /// Retrieve a single stream associated with a particular descriptor
        /// associated with the referenced ticket. A Flight can be composed of one or
        /// more streams where each stream can be retrieved using a separate opaque
        /// ticket that the flight service uses for managing a collection of streams.
        async fn do_get(
            &self,
            request: tonic::Request<super::Ticket>,
        ) -> Result<tonic::Response<Self::DoGetStream>, tonic::Status>;
        ///Server streaming response type for the DoPut method.
        type DoPutStream: futures_core::Stream<Item = Result<super::PutResult, tonic::Status>>
            + Send
            + 'static;
        ///
        /// Push a stream to the flight service associated with a particular
        /// flight stream. This allows a client of a flight service to upload a stream
        /// of data. Depending on the particular flight service, a client consumer
        /// could be allowed to upload a single stream per descriptor or an unlimited
        /// number. In the latter, the service might implement a 'seal' action that
        /// can be applied to a descriptor once all streams are uploaded.
        async fn do_put(
            &self,
            request: tonic::Request<tonic::Streaming<super::FlightData>>,
        ) -> Result<tonic::Response<Self::DoPutStream>, tonic::Status>;
        ///Server streaming response type for the DoExchange method.
        type DoExchangeStream: futures_core::Stream<Item = Result<super::FlightData, tonic::Status>>
            + Send
            + 'static;
        ///
        /// Open a bidirectional data channel for a given descriptor. This
        /// allows clients to send and receive arbitrary Arrow data and
        /// application-specific metadata in a single logical stream. In
        /// contrast to DoGet/DoPut, this is more suited for clients
        /// offloading computation (rather than storage) to a Flight service.
        async fn do_exchange(
            &self,
            request: tonic::Request<tonic::Streaming<super::FlightData>>,
        ) -> Result<tonic::Response<Self::DoExchangeStream>, tonic::Status>;
        ///Server streaming response type for the DoAction method.
        type DoActionStream: futures_core::Stream<Item = Result<super::Result, tonic::Status>>
            + Send
            + 'static;
        ///
        /// Flight services can support an arbitrary number of simple actions in
        /// addition to the possible ListFlights, GetFlightInfo, DoGet, DoPut
        /// operations that are potentially available. DoAction allows a flight client
        /// to do a specific action against a flight service. An action includes
        /// opaque request and response objects that are specific to the type action
        /// being undertaken.
        async fn do_action(
            &self,
            request: tonic::Request<super::Action>,
        ) -> Result<tonic::Response<Self::DoActionStream>, tonic::Status>;
        ///Server streaming response type for the ListActions method.
        type ListActionsStream: futures_core::Stream<Item = Result<super::ActionType, tonic::Status>>
            + Send
            + 'static;
        ///
        /// A flight service exposes all of the available action types that it has
        /// along with descriptions. This allows different flight consumers to
        /// understand the capabilities of the flight service.
        async fn list_actions(
            &self,
            request: tonic::Request<super::Empty>,
        ) -> Result<tonic::Response<Self::ListActionsStream>, tonic::Status>;
    }
    ///
    /// A flight service is an endpoint for retrieving or storing Arrow data. A
    /// flight service can expose one or more predefined endpoints that can be
    /// accessed using the Arrow Flight Protocol. Additionally, a flight service
    /// can expose a set of actions that are available.
    #[derive(Debug)]
    pub struct FlightServiceServer<T: FlightService> {
        inner: _Inner<T>,
        accept_compression_encodings: EnabledCompressionEncodings,
        send_compression_encodings: EnabledCompressionEncodings,
    }
    struct _Inner<T>(Arc<T>);
    impl<T: FlightService> FlightServiceServer<T> {
        pub fn new(inner: T) -> Self {
            Self::from_arc(Arc::new(inner))
        }
        pub fn from_arc(inner: Arc<T>) -> Self {
            let inner = _Inner(inner);
            Self {
                inner,
                accept_compression_encodings: Default::default(),
                send_compression_encodings: Default::default(),
            }
        }
        pub fn with_interceptor<F>(inner: T, interceptor: F) -> InterceptedService<Self, F>
        where
            F: tonic::service::Interceptor,
        {
            InterceptedService::new(Self::new(inner), interceptor)
        }
        /// Enable decompressing requests with the given encoding.
        #[must_use]
        pub fn accept_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.accept_compression_encodings.enable(encoding);
            self
        }
        /// Compress responses with the given encoding, if the client supports it.
        #[must_use]
        pub fn send_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.send_compression_encodings.enable(encoding);
            self
        }
    }
    impl<T, B> tonic::codegen::Service<http::Request<B>> for FlightServiceServer<T>
    where
        T: FlightService,
        B: Body + Send + 'static,
        B::Error: Into<StdError> + Send + 'static,
    {
        type Response = http::Response<tonic::body::BoxBody>;
        type Error = std::convert::Infallible;
        type Future = BoxFuture<Self::Response, Self::Error>;
        fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }
        fn call(&mut self, req: http::Request<B>) -> Self::Future {
            let inner = self.inner.clone();
            match req.uri().path() {
                "/arrow.flight.protocol.FlightService/Handshake" => {
                    #[allow(non_camel_case_types)]
                    struct HandshakeSvc<T: FlightService>(pub Arc<T>);
                    impl<T: FlightService> tonic::server::StreamingService<super::HandshakeRequest>
                        for HandshakeSvc<T>
                    {
                        type Response = super::HandshakeResponse;
                        type ResponseStream = T::HandshakeStream;
                        type Future =
                            BoxFuture<tonic::Response<Self::ResponseStream>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<tonic::Streaming<super::HandshakeRequest>>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move { (*inner).handshake(request).await };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = HandshakeSvc(inner);
                        let codec = tonic::codec::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec).apply_compression_config(
                            accept_compression_encodings,
                            send_compression_encodings,
                        );
                        let res = grpc.streaming(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/arrow.flight.protocol.FlightService/ListFlights" => {
                    #[allow(non_camel_case_types)]
                    struct ListFlightsSvc<T: FlightService>(pub Arc<T>);
                    impl<T: FlightService> tonic::server::ServerStreamingService<super::Criteria>
                        for ListFlightsSvc<T>
                    {
                        type Response = super::FlightInfo;
                        type ResponseStream = T::ListFlightsStream;
                        type Future =
                            BoxFuture<tonic::Response<Self::ResponseStream>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::Criteria>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move { (*inner).list_flights(request).await };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = ListFlightsSvc(inner);
                        let codec = tonic::codec::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec).apply_compression_config(
                            accept_compression_encodings,
                            send_compression_encodings,
                        );
                        let res = grpc.server_streaming(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/arrow.flight.protocol.FlightService/GetFlightInfo" => {
                    #[allow(non_camel_case_types)]
                    struct GetFlightInfoSvc<T: FlightService>(pub Arc<T>);
                    impl<T: FlightService> tonic::server::UnaryService<super::FlightDescriptor>
                        for GetFlightInfoSvc<T>
                    {
                        type Response = super::FlightInfo;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::FlightDescriptor>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move { (*inner).get_flight_info(request).await };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = GetFlightInfoSvc(inner);
                        let codec = tonic::codec::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec).apply_compression_config(
                            accept_compression_encodings,
                            send_compression_encodings,
                        );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/arrow.flight.protocol.FlightService/GetSchema" => {
                    #[allow(non_camel_case_types)]
                    struct GetSchemaSvc<T: FlightService>(pub Arc<T>);
                    impl<T: FlightService> tonic::server::UnaryService<super::FlightDescriptor> for GetSchemaSvc<T> {
                        type Response = super::SchemaResult;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::FlightDescriptor>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move { (*inner).get_schema(request).await };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = GetSchemaSvc(inner);
                        let codec = tonic::codec::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec).apply_compression_config(
                            accept_compression_encodings,
                            send_compression_encodings,
                        );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/arrow.flight.protocol.FlightService/DoGet" => {
                    #[allow(non_camel_case_types)]
                    struct DoGetSvc<T: FlightService>(pub Arc<T>);
                    impl<T: FlightService> tonic::server::ServerStreamingService<super::Ticket> for DoGetSvc<T> {
                        type Response = super::FlightData;
                        type ResponseStream = T::DoGetStream;
                        type Future =
                            BoxFuture<tonic::Response<Self::ResponseStream>, tonic::Status>;
                        fn call(&mut self, request: tonic::Request<super::Ticket>) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move { (*inner).do_get(request).await };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = DoGetSvc(inner);
                        let codec = tonic::codec::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec).apply_compression_config(
                            accept_compression_encodings,
                            send_compression_encodings,
                        );
                        let res = grpc.server_streaming(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/arrow.flight.protocol.FlightService/DoPut" => {
                    #[allow(non_camel_case_types)]
                    struct DoPutSvc<T: FlightService>(pub Arc<T>);
                    impl<T: FlightService> tonic::server::StreamingService<super::FlightData> for DoPutSvc<T> {
                        type Response = super::PutResult;
                        type ResponseStream = T::DoPutStream;
                        type Future =
                            BoxFuture<tonic::Response<Self::ResponseStream>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<tonic::Streaming<super::FlightData>>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move { (*inner).do_put(request).await };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = DoPutSvc(inner);
                        let codec = tonic::codec::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec).apply_compression_config(
                            accept_compression_encodings,
                            send_compression_encodings,
                        );
                        let res = grpc.streaming(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/arrow.flight.protocol.FlightService/DoExchange" => {
                    #[allow(non_camel_case_types)]
                    struct DoExchangeSvc<T: FlightService>(pub Arc<T>);
                    impl<T: FlightService> tonic::server::StreamingService<super::FlightData> for DoExchangeSvc<T> {
                        type Response = super::FlightData;
                        type ResponseStream = T::DoExchangeStream;
                        type Future =
                            BoxFuture<tonic::Response<Self::ResponseStream>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<tonic::Streaming<super::FlightData>>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move { (*inner).do_exchange(request).await };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = DoExchangeSvc(inner);
                        let codec = tonic::codec::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec).apply_compression_config(
                            accept_compression_encodings,
                            send_compression_encodings,
                        );
                        let res = grpc.streaming(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/arrow.flight.protocol.FlightService/DoAction" => {
                    #[allow(non_camel_case_types)]
                    struct DoActionSvc<T: FlightService>(pub Arc<T>);
                    impl<T: FlightService> tonic::server::ServerStreamingService<super::Action> for DoActionSvc<T> {
                        type Response = super::Result;
                        type ResponseStream = T::DoActionStream;
                        type Future =
                            BoxFuture<tonic::Response<Self::ResponseStream>, tonic::Status>;
                        fn call(&mut self, request: tonic::Request<super::Action>) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move { (*inner).do_action(request).await };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = DoActionSvc(inner);
                        let codec = tonic::codec::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec).apply_compression_config(
                            accept_compression_encodings,
                            send_compression_encodings,
                        );
                        let res = grpc.server_streaming(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/arrow.flight.protocol.FlightService/ListActions" => {
                    #[allow(non_camel_case_types)]
                    struct ListActionsSvc<T: FlightService>(pub Arc<T>);
                    impl<T: FlightService> tonic::server::ServerStreamingService<super::Empty> for ListActionsSvc<T> {
                        type Response = super::ActionType;
                        type ResponseStream = T::ListActionsStream;
                        type Future =
                            BoxFuture<tonic::Response<Self::ResponseStream>, tonic::Status>;
                        fn call(&mut self, request: tonic::Request<super::Empty>) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move { (*inner).list_actions(request).await };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = ListActionsSvc(inner);
                        let codec = tonic::codec::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec).apply_compression_config(
                            accept_compression_encodings,
                            send_compression_encodings,
                        );
                        let res = grpc.server_streaming(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                _ => Box::pin(async move {
                    Ok(http::Response::builder()
                        .status(200)
                        .header("grpc-status", "12")
                        .header("content-type", "application/grpc")
                        .body(empty_body())
                        .unwrap())
                }),
            }
        }
    }
    impl<T: FlightService> Clone for FlightServiceServer<T> {
        fn clone(&self) -> Self {
            let inner = self.inner.clone();
            Self {
                inner,
                accept_compression_encodings: self.accept_compression_encodings,
                send_compression_encodings: self.send_compression_encodings,
            }
        }
    }
    impl<T: FlightService> Clone for _Inner<T> {
        fn clone(&self) -> Self {
            Self(self.0.clone())
        }
    }
    impl<T: std::fmt::Debug> std::fmt::Debug for _Inner<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:?}", self.0)
        }
    }
    impl<T: FlightService> tonic::server::NamedService for FlightServiceServer<T> {
        const NAME: &'static str = "arrow.flight.protocol.FlightService";
    }
}
