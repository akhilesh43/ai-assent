"""python -m unittest tests.test_phone_setup -v — the Android manifest patch, on a typical Flutter template."""
import os, sys, unittest
import xml.dom.minidom as minidom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "jarvis_phone"))
import setup_android as sa      # noqa: E402

TEMPLATE = '''<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    <application
        android:label="jarvis_phone"
        android:name="${applicationName}"
        android:icon="@mipmap/ic_launcher">
        <activity
            android:name=".MainActivity"
            android:exported="true"
            android:launchMode="singleTop">
            <intent-filter>
                <action android:name="android.intent.action.MAIN"/>
                <category android:name="android.intent.category.LAUNCHER"/>
            </intent-filter>
        </activity>
        <meta-data android:name="flutterEmbedding" android:value="2" />
    </application>
    <queries>
        <intent>
            <action android:name="android.intent.action.PROCESS_TEXT"/>
            <data android:mimeType="text/plain"/>
        </intent>
    </queries>
</manifest>
'''


class TestManifest(unittest.TestCase):
    def test_patch_is_valid_xml_with_everything_in_place(self):
        out = sa.patch_manifest(TEMPLATE)
        dom = minidom.parseString(out)                       # raises if malformed
        names = {e.getAttribute("android:name") for e in dom.getElementsByTagName("uses-permission")}
        for p in ("ANSWER_PHONE_CALLS", "READ_CONTACTS", "SEND_SMS", "INTERNET", "FOREGROUND_SERVICE_SPECIAL_USE"):
            self.assertIn("android.permission." + p, names)
        comps = {e.getAttribute("android:name") for tag in ("service", "receiver") for e in dom.getElementsByTagName(tag)}
        self.assertEqual(comps, {".BridgeService", ".JarvisListener", ".CallReceiver", ".BootReceiver"})
        app = dom.getElementsByTagName("application")[0]
        self.assertEqual(app.getAttribute("android:usesCleartextTraffic"), "true")
        self.assertEqual(app.getAttribute("android:label"), "JARVIS")
        self.assertEqual(len(dom.getElementsByTagName("activity")), 1)     # the original is untouched

    def test_idempotent(self):
        once = sa.patch_manifest(TEMPLATE)
        self.assertEqual(sa.patch_manifest(once), once)

    def test_rejects_garbage(self):
        with self.assertRaises(ValueError):
            sa.patch_manifest("<nothing/>")

    def test_every_class_in_the_manifest_exists_as_kotlin(self):
        kt = os.path.join(ROOT, "jarvis_phone", "android_overlay", "kotlin")
        for name in ("BridgeService", "JarvisListener", "CallReceiver", "BootReceiver", "MainActivity"):
            self.assertTrue(os.path.exists(os.path.join(kt, name + ".kt")), name)

    def test_phone_actions_match_what_the_laptop_can_send(self):
        from core.phone_bridge import ACTIONS
        src = open(os.path.join(ROOT, "jarvis_phone", "android_overlay", "kotlin", "PhoneActions.kt"), encoding="utf-8").read()
        for a in ACTIONS:
            self.assertIn(f'"{a}"', src, f"phone does not implement {a}")

    def test_dart_endpoints_exist_on_the_server(self):
        dart = open(os.path.join(ROOT, "jarvis_phone", "lib", "api.dart"), encoding="utf-8").read()
        server = open(os.path.join(ROOT, "core", "phone_link.py"), encoding="utf-8").read()
        import re
        for path in set(re.findall(r"'(/api/[a-z\-/]+)'", dart)):
            self.assertIn(f'"{path}"', server, f"server has no {path}")
        kt = open(os.path.join(ROOT, "jarvis_phone", "android_overlay", "kotlin", "BridgeService.kt"), encoding="utf-8").read()
        for path in set(re.findall(r"/api/[a-z\-/]+", kt)):
            self.assertIn(f'"{path}"', server, f"server has no {path}")


if __name__ == "__main__":
    unittest.main()
