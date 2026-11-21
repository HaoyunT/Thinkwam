import resource

def bootstrap():
 soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
 if soft<2048:resource.setrlimit(resource.RLIMIT_NOFILE,(min(2048,hard),hard))

def configure_torch(torch):
 pass
