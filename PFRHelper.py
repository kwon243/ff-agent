import re
from typing import Optional, List, Dict
import unicodedata


class PFRIDMatcher:
    def __init__(self, pfr_database: List[Dict[str, str]]):
        self.database = pfr_database
        self.pfr_ids = [entry['pfr_id'] for entry in pfr_database]

    def normalize_name(self, name: str) -> str:
        name = unicodedata.normalize('NFKD', name)
        name = name.encode('ASCII', 'ignore').decode('ASCII')
        return name

    def extract_name_parts(self, full_name: str) -> tuple:
        full_name = self.normalize_name(full_name.strip())
        full_name = re.sub(r'\s+(Jr\.?|Sr\.?|III?|IV)$', '', full_name, flags=re.IGNORECASE)

        # Handle compound last names like "Amon-Ra St. Brown"
        if ' St. ' in full_name or ' St ' in full_name:
            parts = full_name.split()
            if len(parts) >= 3:
                first_name = ' '.join(parts[:-2])
                last_name = ' '.join(parts[-2:])
                is_initials = bool(re.match(r'^[A-Z]\.(?:[A-Z]\.)?$', first_name.replace(' ', '')))
                return first_name, last_name, is_initials

        # Handle initials (e.g., "T.J. Hockenson", "J.P. Losman")
        initials_match = re.match(r'^([A-Z]\.(?:[A-Z]\.)?)\s+(.+)$', full_name)
        if initials_match:
            first_name = initials_match.group(1)
            last_name = initials_match.group(2)
            return first_name, last_name, True

        # Standard case
        parts = full_name.split()
        if len(parts) >= 2:
            return parts[0], ' '.join(parts[1:]), False
        elif len(parts) == 1:
            return parts[0], '', False
        return '', '', False

    def generate_pfr_patterns(self, first_name: str, last_name: str, is_initials: bool = False) -> List[str]:
        patterns = []
        if not first_name or not last_name:
            return patterns

        if is_initials:
            initials = first_name.replace('.', '')
            initials_no_period = initials.upper()
            initials_with_dot = initials[0].upper() + '.'
            first_parts = [initials_no_period, initials_with_dot]
        else:
            first_clean = first_name.replace(' ', '').replace('-', '').replace("'", "")
            if len(first_clean) >= 2:
                first_parts = [first_clean[0].upper() + first_clean[1].lower()]
            else:
                first_parts = [first_clean[0].upper()]

        if ' ' in last_name:
            # Handle compound last names like "St. Brown"
            last_parts = last_name.split()
            if len(last_parts) == 2:
                first_last = last_parts[0].replace('.', '')
                second_last = last_parts[1]

                # Pattern: St.B + First2 → "St.BEq00"
                compound_pattern = first_last.capitalize() + '.' + second_last[0].upper()
                for f in first_parts:
                    patterns.append(compound_pattern + f)

                # Pattern: Stxx + First2 → "StxxAm00"
                if first_last.lower() == 'st':
                    stxx_pattern = 'Stxx'
                    for f in first_parts:
                        patterns.append(stxx_pattern + f)

                # Fallback: standard 4+2 combo with spaces removed
                last_clean = (last_parts[0] + last_parts[1]).replace('.', '').replace(' ', '')
                if len(last_clean) <= 4:
                    last_part = last_clean.capitalize().ljust(4, 'x')
                else:
                    last_part = last_clean[0].upper() + last_clean[1:4].lower()
                for f in first_parts:
                    patterns.append(last_part + f)
        else:
            # Non-compound last names
            last_clean = last_name.replace('.', '').replace(' ', '').replace('-', '').replace("'", "")
            if len(last_clean) <= 4:
                last_part = last_clean.capitalize().ljust(4, 'x')
            else:
                last_part = last_clean[0].upper() + last_clean[1:4].lower()
            for f in first_parts:
                patterns.append(last_part + f)

        return patterns

    def find_matching_ids(self, player_name: str) -> List[str]:
        first_name, last_name, is_initials = self.extract_name_parts(player_name)
        if not first_name or not last_name:
            return []

        patterns = self.generate_pfr_patterns(first_name, last_name, is_initials)
        matches = []

        for pfr_id in self.pfr_ids:
            base_id = pfr_id[:-2] if len(pfr_id) >= 2 and pfr_id[-2:].isdigit() else pfr_id
            for pattern in patterns:
                if base_id.lower() == pattern.lower():
                    matches.append(pfr_id)
                    break
        return matches

    def get_pfr_id(self, player_name: str) -> Optional[str]:
        matches = self.find_matching_ids(player_name)
        return matches[0] if len(matches) == 1 else None

    def get_pfr_id_with_info(self, player_name: str) -> Dict:
        matches = self.find_matching_ids(player_name)
        return {
            'pfr_id': matches[0] if len(matches) == 1 else None,
            'match_count': len(matches),
            'all_matches': matches,
            'status': 'unique' if len(matches) == 1 else ('multiple' if len(matches) > 1 else 'no_match')
        }


if __name__ == "__main__":
    sample_database = [
        {'pfr_id': 'BarkSa00'},
        {'pfr_id': 'GibbJa00'},
        {'pfr_id': 'GibbJa01'},
        {'pfr_id': 'BrowMa04'},
        {'pfr_id': 'StxxAm00'},
        {'pfr_id': 'HockTJ00'},
        {'pfr_id': 'St.BEq00'},
        {'pfr_id': 'NixxBo00'},
        {'pfr_id': 'LosmJ.00'},
        {'pfr_id': 'YateT.00'},
        {'pfr_id': 'FeelA.00'},
        {'pfr_id': 'OConAi00'}
    ]

    matcher = PFRIDMatcher(sample_database)

    test_names = [
        "Saquon Barkley",
        "Jahmyr Gibbs",
        "Marquise Brown",
        "Amon-Ra St. Brown",
        "T.J. Hockenson",
        "Equanimeous St. Brown",
        "Bo Nix",
        "J.P. Losman",
        "T.J. Yates",
        "A.J. Feeley",
        "Aidan O'Connell"
    ]

    for name in test_names:
        result = matcher.get_pfr_id_with_info(name)
        print(f"\nPlayer: {name}")
        print(f"  Status: {result['status']}")
        print(f"  Matches found: {result['match_count']}")
        if result['all_matches']:
            print(f"  PFR IDs: {', '.join(result['all_matches'])}")
        if result['pfr_id']:
            print(f"  Selected ID: {result['pfr_id']}")
