"""Base implementation of 0MQ authentication."""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

import logging

import zmq
from zmq.utils import z85
from zmq.utils.strtypes import bytes, unicode, b, u
from zmq.error import _check_version

from .certs import load_certificates


CURVE_ALLOW_ANY = '*'
VERSION = b'1.0'

class Authenticator(object):
    """Implementation of ZAP authentication for zmq connections.

    Note:

    - libzmq provides four levels of security: default NULL (which the Authenticator does
      not see), and authenticated NULL, PLAIN, CURVE, and GSSAPI, which the Authenticator can see.
    - until you add policies, all incoming NULL connections are allowed.
      (classic ZeroMQ behavior), and all PLAIN and CURVE connections are denied.
    - GSSAPI requires no configuration.
    """

    def __init__(self, context=None, encoding='utf-8', log=None):
        _check_version((4,0), "security")
        self.context = context or zmq.Context.instance()
        self.encoding = encoding
        self.allow_any = False
        self.credentials_providers = {}
        self.zap_socket = None
        self.whitelist = set()
        self.blacklist = set()
        # passwords is a dict keyed by domain and contains values
        # of dicts with username:password pairs.
        self.passwords = {}
        # certs is dict keyed by domain and contains values
        # of dicts keyed by the public keys from the specified location.
        self.certs = {}
        self.log = log or logging.getLogger('zmq.auth')
    
    def start(self):
        """Create and bind the ZAP socket"""
        self.zap_socket = self.context.socket(zmq.REP)
        self.zap_socket.linger = 1
        self.zap_socket.bind("inproc://zeromq.zap.01")
        self.log.debug("Starting")

    def stop(self):
        """Close the ZAP socket"""
        if self.zap_socket:
            self.zap_socket.close()
        self.zap_socket = None

    def allow(self, *addresses):
        """Allow (whitelist) IP address(es).
        
        Connections from addresses not in the whitelist will be rejected.
        
        - For NULL, all clients from this address will be accepted.
        - For real auth setups, they will be allowed to continue with authentication.
        
        whitelist is mutually exclusive with blacklist.
        """
        if self.blacklist:
            raise ValueError("Only use a whitelist or a blacklist, not both")
        self.log.debug("Allowing %s", ','.join(addresses))
        self.whitelist.update(addresses)

    def deny(self, *addresses):
        """Deny (blacklist) IP address(es).
        
        Addresses not in the blacklist will be allowed to continue with authentication.
        
        Blacklist is mutually exclusive with whitelist.
        """
        if self.whitelist:
            raise ValueError("Only use a whitelist or a blacklist, not both")
        self.log.debug("Denying %s", ','.join(addresses))
        self.blacklist.update(addresses)

    def configure_plain(self, domain='*', passwords=None):
        """Configure PLAIN authentication for a given domain.
        
        PLAIN authentication uses a plain-text password file.
        To cover all domains, use "*".
        You can modify the password file at any time; it is reloaded automatically.
        """
        if passwords:
            self.passwords[domain] = passwords
        self.log.debug("Configure plain: %s", domain)

    def configure_curve(self, domain='*', location=None):
        """Configure CURVE authentication for a given domain.
        
        CURVE authentication uses a directory that holds all public client certificates,
        i.e. their public keys.
        
        To cover all domains, use "*".
        
        You can add and remove certificates in that directory at any time. configure_curve must be called 
        every time certificates are added or removed, in order to update the Authenticator's state 
        
        To allow all client keys without checking, specify CURVE_ALLOW_ANY for the location.
        """
        # If location is CURVE_ALLOW_ANY then allow all clients. Otherwise
        # treat location as a directory that holds the certificates.
        self.log.debug("Configure curve: %s[%s]", domain, location)
        if location == CURVE_ALLOW_ANY:
            self.allow_any = True
        else:
            self.allow_any = False
            try:
                self.certs[domain] = load_certificates(location)
            except Exception as e:
                self.log.error("Failed to load CURVE certs from %s: %s", location, e)

    def configure_curve_callback(self, domain='*', credentials_provider=None):
        """Configure CURVE authentication for a given domain.

        CURVE authentication using a callback function validating
        the client public key according to a custom mechanism, e.g. checking the
        key against records in a db. credentials_provider is an object of a class which
        implements a callback method accepting two parameters (domain and key), e.g.::

            class CredentialsProvider(object):

                def __init__(self):
                    ...e.g. db connection

                def callback(self, domain, key):
                    valid = ...lookup key and/or domain in db
                    if valid:
                        logging.info('Authorizing: {0}, {1}'.format(domain, key))
                        return True
                    else:
                        logging.warning('NOT Authorizing: {0}, {1}'.format(domain, key))
                        return False

        To cover all domains, use "*".

        To allow all client keys without checking, specify CURVE_ALLOW_ANY for the location.
        """

        self.allow_any = False

        if credentials_provider is not None:
            self.credentials_providers[domain] = credentials_provider
        else:
            self.log.error("None credentials_provider provided for domain:%s",domain)

    def curve_user_id(self, client_public_key):
        """Return the User-Id corresponding to a CURVE client's public key
        
        Default implementation uses the z85-encoding of the public key.
        
        Override to define a custom mapping of public key : user-id
        
        This is only called on successful authentication.
        
        Parameters
        ----------
        client_public_key: bytes
            The client public key used for the given message
        
        Returns
        -------
        user_id: unicode
            The user ID as text
        """
        return z85.encode(client_public_key).decode('ascii')

    def configure_gssapi(self, domain='*', location=None):
        """Configure GSSAPI authentication
        
        Currently this is a no-op because there is nothing to configure with GSSAPI.
        """
        pass

    def handle_zap_message(self, msg):
        """Perform ZAP authentication"""
        if len(msg) < 6:
            self.log.error("Invalid ZAP message, not enough frames: %r", msg)
            if len(msg) < 2:
                self.log.error("Not enough information to reply")
            else:
                self._send_zap_reply(msg[1], b"400", b"Not enough frames")
            return
        
        version, request_id, domain, address, identity, mechanism = msg[:6]
        credentials = msg[6:]
        
        domain = u(domain, self.encoding, 'replace')
        address = u(address, self.encoding, 'replace')

        if (version != VERSION):
            self.log.error("Invalid ZAP version: %r", msg)
            self._send_zap_reply(request_id, b"400", b"Invalid version")
            return

        self.log.debug("version: %r, request_id: %r, domain: %r,"
                      " address: %r, identity: %r, mechanism: %r",
                      version, request_id, domain,
                      address, identity, mechanism,
        )


        # Is address is explicitly whitelisted or blacklisted?
        allowed = False
        denied = False
        reason = b"NO ACCESS"

        if self.whitelist:
            if address in self.whitelist:
                allowed = True
                self.log.debug("PASSED (whitelist) address=%s", address)
            else:
                denied = True
                reason = b"Address not in whitelist"
                self.log.debug("DENIED (not in whitelist) address=%s", address)

        elif self.blacklist:
            if address in self.blacklist:
                denied = True
                reason = b"Address is blacklisted"
                self.log.debug("DENIED (blacklist) address=%s", address)
            else:
                allowed = True
                self.log.debug("PASSED (not in blacklist) address=%s", address)

        # Perform authentication mechanism-specific checks if necessary
        username = u("anonymous")
        if not denied:

            if mechanism == b'NULL' and not allowed:
                # For NULL, we allow if the address wasn't blacklisted
                self.log.debug("ALLOWED (NULL)")
                allowed = True

            elif mechanism == b'PLAIN':
                # For PLAIN, even a whitelisted address must authenticate
                if len(credentials) != 2:
                    self.log.error("Invalid PLAIN credentials: %r", credentials)
                    self._send_zap_reply(request_id, b"400", b"Invalid credentials")
                    return
                username, password = [ u(c, self.encoding, 'replace') for c in credentials ]
                allowed, reason = self._authenticate_plain(domain, username, password)

            elif mechanism == b'CURVE':
                # For CURVE, even a whitelisted address must authenticate
                if len(credentials) != 1:
                    self.log.error("Invalid CURVE credentials: %r", credentials)
                    self._send_zap_reply(request_id, b"400", b"Invalid credentials")
                    return
                key = credentials[0]
                allowed, reason = self._authenticate_curve(domain, key)
                if allowed:
                    username = self.curve_user_id(key)
                    
            elif mechanism == b'GSSAPI':
                if len(credentials) != 1:
                    self.log.error("Invalid GSSAPI credentials: %r", credentials)
                    self._send_zap_reply(request_id, b"400", b"Invalid credentials")
                    return
                # use principal as user-id for now
                principal = username = credentials[0]
                allowed, reason = self._authenticate_gssapi(domain, principal)

        if allowed:
            self._send_zap_reply(request_id, b"200", b"OK", username)
        else:
            self._send_zap_reply(request_id, b"400", reason)

    def _authenticate_plain(self, domain, username, password):
        """PLAIN ZAP authentication"""
        allowed = False
        reason = b""
        if self.passwords:
            # If no domain is not specified then use the default domain
            if not domain:
                domain = '*'

            if domain in self.passwords:
                if username in self.passwords[domain]:
                    if password == self.passwords[domain][username]:
                        allowed = True
                    else:
                        reason = b"Invalid password"
                else:
                    reason = b"Invalid username"
            else:
                reason = b"Invalid domain"

            if allowed:
                self.log.debug("ALLOWED (PLAIN) domain=%s username=%s password=%s",
                    domain, username, password,
                )
            else:
                self.log.debug("DENIED %s", reason)

        else:
            reason = b"No passwords defined"
            self.log.debug("DENIED (PLAIN) %s", reason)

        return allowed, reason

    def _authenticate_curve(self, domain, client_key):
        """CURVE ZAP authentication"""
        allowed = False
        reason = b""
        if self.allow_any:
            allowed = True
            reason = b"OK"
            self.log.debug("ALLOWED (CURVE allow any client)")
        elif self.credentials_providers != {}:
            # If no explicit domain is specified then use the default domain
            if not domain:
                domain = '*'

            if domain in self.credentials_providers:
                z85_client_key = z85.encode(client_key)
                # Callback to check if key is Allowed
                if (self.credentials_providers[domain].callback(domain, z85_client_key)):
                    allowed = True
                    reason = b"OK"
                else:
                    reason = b"Unknown key"

                status = "ALLOWED" if allowed else "DENIED"
                self.log.debug("%s (CURVE auth_callback) domain=%s client_key=%s",
                    status, domain, z85_client_key,
                )
            else:
                reason = b"Unknown domain"
        else:
            # If no explicit domain is specified then use the default domain
            if not domain:
                domain = '*'

            if domain in self.certs:
                # The certs dict stores keys in z85 format, convert binary key to z85 bytes
                z85_client_key = z85.encode(client_key)
                if self.certs[domain].get(z85_client_key):
                    allowed = True
                    reason = b"OK"
                else:
                    reason = b"Unknown key"

                status = "ALLOWED" if allowed else "DENIED"
                self.log.debug("%s (CURVE) domain=%s client_key=%s",
                    status, domain, z85_client_key,
                )
            else:
                reason = b"Unknown domain"

        return allowed, reason

    def _authenticate_gssapi(self, domain, principal):
        """Nothing to do for GSSAPI, which has already been handled by an external service."""
        self.log.debug("ALLOWED (GSSAPI) domain=%s principal=%s", domain, principal)
        return True, b'OK'

    def _send_zap_reply(self, request_id, status_code, status_text, user_id='anonymous'):
        """Send a ZAP reply to finish the authentication."""
        user_id = user_id if status_code == b'200' else b''
        if isinstance(user_id, unicode):
            user_id = user_id.encode(self.encoding, 'replace')
        metadata = b''  # not currently used
        self.log.debug("ZAP reply code=%s text=%s", status_code, status_text)
        reply = [VERSION, request_id, status_code, status_text, user_id, metadata]
        self.zap_socket.send_multipart(reply)

__all__ = ['Authenticator', 'CURVE_ALLOW_ANY']
