"""python -m unittest tests.test_phone_doctor -v"""
import os, socket, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phone_doctor as pd   # noqa: E402


class TestDoctor(unittest.TestCase):
    def test_pair_url_matches_what_the_app_parses(self):
        u = pd.build_pair_url("192.168.1.5", "ABCD1234", ["10.0.0.4", "172.16.0.2"])
        self.assertEqual(u, "http://192.168.1.5:8765/pair?key=ABCD1234&alt=10.0.0.4,172.16.0.2")
        self.assertEqual(pd.build_pair_url("192.168.1.5", "K"), "http://192.168.1.5:8765/pair?key=K")

    def test_port_probe(self):
        s = socket.socket(); s.bind(("0.0.0.0", 0)); s.listen(1)
        port = s.getsockname()[1]
        try:
            self.assertTrue(pd.port_in_use(port))
        finally:
            s.close()
        self.assertFalse(pd.port_in_use(port))

    def test_ping_of_nothing_is_none(self):
        self.assertIsNone(pd.ping("127.0.0.1", 1, timeout=0.3))

    def test_report_runs_and_explains(self):
        lines = []
        pd.report(lines.append)
        text = "\n".join(lines)
        self.assertIn("checking this laptop", text); self.assertIn("same Wi-Fi", text)

    def test_lan_ips_are_private_and_unique(self):
        from core.phone_link import lan_ips
        got = lan_ips()
        self.assertEqual(len(got), len(set(got)))
        for ip in got:
            a, b = map(int, ip.split(".")[:2])
            self.assertTrue(a == 10 or (a == 172 and 16 <= b <= 31) or (a == 192 and b == 168), ip)


if __name__ == "__main__":
    unittest.main()
