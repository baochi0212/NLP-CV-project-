"""
 # Copyright (c) 2022, salesforce.com, inc.
 # All rights reserved.
 # SPDX-License-Identifier: BSD-3-Clause
 # For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from app.multipage import MultiPage
from app import text_localization as tl
from app import vqa


if __name__ == "__main__":
    app = MultiPage()

    app.add_page("Visual Question Answering", vqa.app)
    app.add_page("Text Localization", tl.app)
    
    app.run()
