Convert Timezone
---
```
from datetime import datetime, timezone
import pytz

x = "2020-10-18 18:25:20.6000"
d = datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)

d.astimezone(tz=pytz.timezone("Australia/Brisbane"))
```
