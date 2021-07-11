#ifdef WIN32
#ifndef DBG_NEW
#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
#endif
#else
#define DBG_NEW new
#endif

