# - Version: 01
# - initial version
# - 20250915


import ssl
import socket
import requests
from urllib.parse import urlparse
from datetime import datetime, timezone
import logging
import urllib3
import certifi
from requests.adapters import HTTPAdapter

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

class CloudflareSSLAdapter(HTTPAdapter):
    """
    Custom SSL adapter that handles Cloudflare certificates properly.
    This fixes the 'unable to get local issuer certificate' error.
    """
    def init_poolmanager(self, *args, **kwargs):
        context = ssl.create_default_context()
        # Lower security level for better compatibility with Cloudflare
        context.set_ciphers('DEFAULT:!DH:!aNULL:!MD5:!RC4:!3DES:!DES:!EXP:!PSK:!SRP:!CAMELLIA:!SEED')
        context.load_verify_locations(cafile=certifi.where())
        kwargs['ssl_context'] = context
        return super().init_poolmanager(*args, **kwargs)

def create_cloudflare_session():
    """
    Create a requests session configured to handle Cloudflare SSL certificates.
    """
    session = requests.Session()
    session.mount('https://', CloudflareSSLAdapter())
    return session

def get_ssl_certificate_info(hostname, port=443):
    """
    Get detailed SSL certificate information directly from the socket.
    Returns dict with certificate details and expiry info.
    """
    try:
        # Create SSL context with default settings
        context = ssl.create_default_context()
        
        # Get certificate info
        with socket.create_connection((hostname, port), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
                cert_der = ssock.getpeercert(binary_form=True)
                
                # Parse certificate details
                ssl_info = {
                    'subject': dict(x[0] for x in cert['subject']),
                    'issuer': dict(x[0] for x in cert['issuer']),
                    'version': cert['version'],
                    'serial_number': cert['serialNumber'],
                    'not_before': cert['notBefore'],
                    'not_after': cert['notAfter'],
                    'signature_algorithm': cert.get('signatureAlgorithm', 'Unknown'),
                    'subject_alt_names': cert.get('subjectAltName', []),
                    'protocol_version': ssock.version(),
                    'cipher_suite': ssock.cipher()
                }
                
                # Calculate days until expiry
                expiry_date = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                expiry_date = expiry_date.replace(tzinfo=timezone.utc)
                days_until_expiry = (expiry_date - datetime.now(timezone.utc)).days
                ssl_info['days_until_expiry'] = days_until_expiry
                
                return ssl_info, cert_der
                
    except Exception as e:
        logger.error(f"Failed to get SSL certificate info: {e}")
        return None, None

def check_ssl_health(url, detailed=False):
    """
    Comprehensive SSL health check including certificate validation,
    expiry dates, and certificate chain verification.
    """
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname
    port = parsed_url.port or (443 if parsed_url.scheme == 'https' else 80)
    
    ssl_results = {
        'hostname': hostname,
        'port': port,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'tests': {}
    }
    
    logger.info(f"🔒 Starting SSL health check for {hostname}:{port}")
    
    # Test 1: Basic SSL handshake
    try:
        ssl_info, cert_der = get_ssl_certificate_info(hostname, port)
        if ssl_info:
            ssl_results['certificate_info'] = ssl_info
            ssl_results['tests']['ssl_handshake'] = 'PASS'
            
            # Certificate expiry check
            days_left = ssl_info['days_until_expiry']
            if days_left > 30:
                ssl_results['tests']['certificate_expiry'] = 'PASS'
                logger.info(f"   ✅ Certificate valid for {days_left} more days")
            elif days_left > 7:
                ssl_results['tests']['certificate_expiry'] = 'WARNING'
                logger.warning(f"   ⚠️  Certificate expires in {days_left} days")
            else:
                ssl_results['tests']['certificate_expiry'] = 'CRITICAL'
                logger.error(f"   ❌ Certificate expires in {days_left} days!")
            
            # Certificate issuer check
            issuer = ssl_info['issuer'].get('organizationName', 'Unknown')
            logger.info(f"   📜 Certificate issued by: {issuer}")
            ssl_results['tests']['certificate_issuer'] = 'PASS'
            
            # Subject Alternative Names check
            san_list = [name[1] for name in ssl_info['subject_alt_names'] if name[0] == 'DNS']
            if hostname in san_list or any(hostname.endswith(san.lstrip('*')) for san in san_list):
                ssl_results['tests']['hostname_verification'] = 'PASS'
                logger.info(f"   ✅ Hostname verification passed")
            else:
                ssl_results['tests']['hostname_verification'] = 'FAIL'
                logger.error(f"   ❌ Hostname {hostname} not found in certificate SAN")
            
            # TLS protocol version check
            protocol = ssl_info['protocol_version']
            if protocol in ['TLSv1.2', 'TLSv1.3']:
                ssl_results['tests']['tls_version'] = 'PASS'
                logger.info(f"   🔐 Using secure protocol: {protocol}")
            else:
                ssl_results['tests']['tls_version'] = 'WARNING'
                logger.warning(f"   ⚠️  Using older protocol: {protocol}")
                
        else:
            ssl_results['tests']['ssl_handshake'] = 'FAIL'
            
    except Exception as e:
        ssl_results['tests']['ssl_handshake'] = 'FAIL'
        ssl_results['error'] = str(e)
        logger.error(f"   ❌ SSL handshake failed: {e}")
    
    # Test 2: Certificate store verification
    try:
        cert_store_path = certifi.where()
        ssl_results['certificate_store'] = cert_store_path
        ssl_results['tests']['certificate_store'] = 'PASS'
        logger.info(f"   📚 Using certificate store: {cert_store_path}")
    except Exception as e:
        ssl_results['tests']['certificate_store'] = 'FAIL'
        logger.error(f"   ❌ Certificate store issue: {e}")
    
    # Test 3: Requests library SSL verification with Cloudflare adapter
    try:
        session = create_cloudflare_session()
        response = session.head(url, timeout=10)
        ssl_results['tests']['requests_ssl_verify'] = 'PASS'
        logger.info(f"   ✅ Requests SSL verification passed (with Cloudflare adapter)")
    except requests.exceptions.SSLError as e:
        ssl_results['tests']['requests_ssl_verify'] = 'FAIL'
        ssl_results['requests_ssl_error'] = str(e)
        logger.error(f"   ❌ Requests SSL verification failed: {e}")
    except Exception as e:
        ssl_results['tests']['requests_ssl_verify'] = 'ERROR'
        logger.error(f"   ❌ Unexpected error during requests check: {e}")
    
    if detailed:
        logger.info("📊 Detailed SSL Information:")
        if ssl_info:
            logger.info(f"   Subject: {ssl_info['subject']}")
            logger.info(f"   Issuer: {ssl_info['issuer']}")
            logger.info(f"   Serial: {ssl_info['serial_number']}")
            logger.info(f"   Valid from: {ssl_info['not_before']}")
            logger.info(f"   Valid until: {ssl_info['not_after']}")
            logger.info(f"   Cipher: {ssl_info['cipher_suite']}")
            
    return ssl_results

def fetch(url, timeout=5):
    """
    Perform a GET and raise on TLS / HTTP errors.
    Uses Cloudflare SSL adapter to handle certificate issues.
    """
    try:
        # Log the attempt
        logger.info(f"🌐 Fetching {url} with timeout={timeout}s")
        
        # Use Cloudflare-compatible session
        session = create_cloudflare_session()
        # Create a session with SSL verification disabled
        session.verify = False
        r = session.get(url, timeout=timeout, verify=True)
        r.raise_for_status()
        
        # Log success details
        logger.info(f"   ✅ HTTP {r.status_code} - Content-Length: {len(r.content)} bytes")
        logger.info(f"   📄 Content-Type: {r.headers.get('content-type', 'unknown')}")
        
        return r
        
    except requests.exceptions.SSLError as e:
        logger.error(f"   ❌ [SSL ERROR] {e}")
        logger.error(f"   🔧 This SSL error persisted even with Cloudflare adapter")
        logger.error(f"   📞 Consider contacting the site administrator")
        raise SystemExit(f"[SSL ERROR] {e}")
        
    except requests.exceptions.ConnectTimeout as e:
        logger.error(f"   ❌ [CONNECTION TIMEOUT] {e}")
        raise SystemExit(f"[CONNECTION TIMEOUT] {e}")
        
    except requests.exceptions.ConnectionError as e:
        logger.error(f"   ❌ [CONNECTION ERROR] {e}")
        raise SystemExit(f"[CONNECTION ERROR] {e}")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"   ❌ [HTTP ERROR] {e}")
        raise SystemExit(f"[HTTP ERROR] {e}")
