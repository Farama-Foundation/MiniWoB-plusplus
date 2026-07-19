"""Environments for MiniWoB++ tasks."""

from miniwob.environment import MiniWoBEnvironment


class AscendingNumbersEnv(MiniWoBEnvironment):
    """
    ## Description

    Click on the numbers in ascending order.

    ## Example utterances

    * Click on the numbers in ascending order.

    ## Utterance fields

    (none)

    ## Additional notes

    * **Partial reward:** If the first number is correctly clicked, the partial reward is 1 - fraction of remaining numbers.
    """

    subdomain = "ascending-numbers"


class BisectAngleEnv(MiniWoBEnvironment):
    """
    ## Description

    Find the line that bisects an angle evenly in two.

    ## Example utterances

    * Create a line that bisects the angle evenly in two, then press submit.

    ## Utterance fields

    (none)

    ## Additional notes

    * **Partial reward:** If the line is inside the angle, the partial reward is how close it is to the bisector (from 0 to 1).
    * Since the reward is a continuous value, use the `get_thresholded_reward` reward preprocessor to binarize the reward instead of `get_binary_reward`.
    """

    subdomain = "bisect-angle"


class BookFlightEnv(MiniWoBEnvironment):
    """
    ## Description

    Search for flight results.

    ## Example utterances

    * Book the cheapest one-way flight from: Anvik, AK to: NYC on 12/23/2016.
    * Book the cheapest one-way flight from: HIB to: SWD on 10/25/2016.
    * Book the cheapest one-way flight from: HKB to: Decatur, IL on 10/29/2016.
    * Book the shortest one-way flight from: Cincinnati, OH to: LEX on 10/16/2016.
    * Book the shortest one-way flight from: RAP to: ICT on 11/16/2016.

    ## Utterance fields

    * criterion
    * date
    * from
    * to
    """

    subdomain = "book-flight"


class BookFlightNodelayEnv(MiniWoBEnvironment):
    """
    ## Description

    [book-flight] Removed animation.

    ## Example utterances

    * Book the cheapest one-way flight from: Anvik, AK to: NYC on 12/23/2016.
    * Book the cheapest one-way flight from: HIB to: SWD on 10/25/2016.
    * Book the cheapest one-way flight from: HKB to: Decatur, IL on 10/29/2016.
    * Book the shortest one-way flight from: Cincinnati, OH to: LEX on 10/16/2016.
    * Book the shortest one-way flight from: RAP to: ICT on 11/16/2016.

    ## Utterance fields

    * criterion
    * date
    * from
    * to
    """

    subdomain = "book-flight-nodelay"


class BuyTicketEnv(MiniWoBEnvironment):
    """
    ## Description

    Buy a ticket that matches the requested criteria.

    ## Example utterances

    * Buy the ticket with the cheapest cost.
    * Buy the ticket with the longest duration.
    * Buy the ticket with the most expensive cost.
    * Buy the ticket with the shortest duration.

    ## Utterance fields

    * target

    ## Additional notes

    * **Partial reward:** If the bought ticket is not the best but also not the worst according to the criterion, the partial reward is -0.5.
    """

    subdomain = "buy-ticket"


class ChooseDateEnv(MiniWoBEnvironment):
    """
    ## Description

    Learn to operate a date picker tool.

    ## Example utterances

    * Select 01/14/2016 as the date and hit submit.
    * Select 03/05/2016 as the date and hit submit.
    * Select 05/15/2016 as the date and hit submit.
    * Select 05/17/2016 as the date and hit submit.
    * Select 10/03/2016 as the date and hit submit.

    ## Utterance fields

    * day
    * month
    * year
    """

    subdomain = "choose-date"


class ChooseDateEasyEnv(MiniWoBEnvironment):
    """
    ## Description

    [choose-date] December only.

    ## Example utterances

    * Select 12/02/2016 as the date and hit submit.
    * Select 12/06/2016 as the date and hit submit.
    * Select 12/12/2016 as the date and hit submit.
    * Select 12/23/2016 as the date and hit submit.
    * Select 12/27/2016 as the date and hit submit.

    ## Utterance fields

    * day
    * month
    * year
    """

    subdomain = "choose-date-easy"


class ChooseDateMediumEnv(MiniWoBEnvironment):
    """
    ## Description

    [choose-date] December or November only.

    ## Example utterances

    * Select 11/03/2016 as the date and hit submit.
    * Select 11/11/2016 as the date and hit submit.
    * Select 11/23/2016 as the date and hit submit.
    * Select 12/16/2016 as the date and hit submit.
    * Select 12/24/2016 as the date and hit submit.

    ## Utterance fields

    * day
    * month
    * year
    """

    subdomain = "choose-date-medium"


class ChooseDateNodelayEnv(MiniWoBEnvironment):
    """
    ## Description

    [choose-date] Removed animation.

    ## Example utterances

    * Select 01/14/2016 as the date and hit submit.
    * Select 03/05/2016 as the date and hit submit.
    * Select 05/15/2016 as the date and hit submit.
    * Select 05/17/2016 as the date and hit submit.
    * Select 10/03/2016 as the date and hit submit.

    ## Utterance fields

    * day
    * month
    * year
    """

    subdomain = "choose-date-nodelay"


class ChooseListEnv(MiniWoBEnvironment):
    """
    ## Description

    Choose an item from a drop down list.

    ## Example utterances

    * Select Betty from the list and click Submit.
    * Select Bobine from the list and click Submit.
    * Select Ferne from the list and click Submit.
    * Select Heard Island and McDonald Islands from the list and click Submit.
    * Select Helli from the list and click Submit.

    ## Utterance fields

    * target
    """

    subdomain = "choose-list"


class CircleCenterEnv(MiniWoBEnvironment):
    """
    ## Description

    Find the center of a circle.

    ## Example utterances

    * Find and click on the center of the circle, then press submit.

    ## Utterance fields

    (none)

    ## Additional notes

    * **Partial reward:** If the point is inside the circle, the partial reward is how close it is to the center (from 0 to 1).
    * Since the reward is a continuous value, use the `get_thresholded_reward` reward preprocessor to binarize the reward instead of `get_binary_reward`.
    """

    subdomain = "circle-center"


class ClickButtonEnv(MiniWoBEnvironment):
    """
    ## Description

    Click on a specific button in a generated form.

    ## Example utterances

    * Click on the "Ok" button.
    * Click on the "no" button.
    * Click on the "ok" button.
    * Click on the "okay" button.
    * Click on the "submit" button.

    ## Utterance fields

    * target

    ## Additional notes

    * Clicking other non-buttons first is OK.
    """

    subdomain = "click-button"


class ClickButtonSequenceEnv(MiniWoBEnvironment):
    """
    ## Description

    Click on buttons in a certain order.

    ## Example utterances

    * Click button ONE, then click button TWO.

    ## Utterance fields

    (none)
    """

    subdomain = "click-button-sequence"


class ClickCheckboxesEnv(MiniWoBEnvironment):
    """
    ## Description

    Click desired checkboxes.

    ## Example utterances

    * Select 91YPF, i6Vdpn2, nd7Qt, XPMut and click Submit.
    * Select DKkQH and click Submit.
    * Select HF2 and click Submit.
    * Select fzzqo, NYYyS82 and click Submit.
    * Select nothing and click Submit.

    ## Utterance fields

    * button
    * target 0
    * target 1
    * target 2
    * target 3

    ## Additional notes

    * **Partial reward:** The score is 1 for each correct checkbox and -1 for each incorrect one. The reward is the average score.
    """

    subdomain = "click-checkboxes"


class ClickCheckboxesLargeEnv(MiniWoBEnvironment):
    """
    ## Description

    [click-checkboxes] Click at least 5 out of up to 12 checkboxes.

    ## Example utterances

    * Select 4aycHnY, 1YPFe2i, gbQ4S2, Vdpn2dP, OX, Mutiv, d7Qt, fwDwN, ZH2tL1, UJWGA, pRdsyR and click Submit.
    * Select 7jRPftZ, 5WZHv, TvFgM, yZZUALj, nXH, EoPV, qtdjmII, Q0af and click Submit.
    * Select k9, Kh, zzq, key, bA2, gZf3, nhG97 and click Submit.
    * Select kQHh8j, SaIGD, wjy8, 4tF, oB, 2uMiJWr and click Submit.
    * Select nb, 8qiSp3m, pKybC69, v8Zs3, U6gHF and click Submit.

    ## Utterance fields

    * button
    * target 0
    * target 1
    * target 10
    * target 11
    * target 2
    * target 3
    * target 4
    * target 5
    * target 6
    * target 7
    * target 8
    * target 9

    ## Additional notes

    * **Partial reward:** The score is 1 for each correct checkbox and -1 for each incorrect one. The reward is the average score.
    """

    subdomain = "click-checkboxes-large"


class ClickCheckboxesSoftEnv(MiniWoBEnvironment):
    """
    ## Description

    [click-checkboxes] Paraphrased entries.

    ## Example utterances

    * Select words similar to brave, chubby, depraved and click Submit.
    * Select words similar to gleeful, swine, rabbits, flames and click Submit.
    * Select words similar to mild, delicious and click Submit.
    * Select words similar to red and click Submit.
    * Select words similar to response, panicked and click Submit.

    ## Utterance fields

    * button
    * target 0
    * target 1
    * target 2
    * target 3
    * target 4

    ## Additional notes

    * **Partial reward:** The score is 1 for each correct checkbox and -1 for each incorrect one. The reward is the average score.
    """

    subdomain = "click-checkboxes-soft"


class ClickCheckboxesTransferEnv(MiniWoBEnvironment):
    """
    ## Description

    [click-checkboxes] Train and test on different number of targets.

    ## Example utterances

    * Select FBYPK4g, coN4Zj, ICsb and click Submit.
    * Select Mutiv, d7Qt, OX and click Submit.
    * Select ZV6g and click Submit.
    * Select k9 and click Submit.
    * Select nothing and click Submit.

    ## Utterance fields

    * button
    * target 0
    * target 1
    * target 2

    ## Additional notes

    * **Partial reward:** The score is 1 for each correct checkbox and -1 for each incorrect one. The reward is the average score.
    * Use the `set_data_mode` method of the environment to switch between the train and test scenarios.
    """

    subdomain = "click-checkboxes-transfer"


class ClickCollapsibleEnv(MiniWoBEnvironment):
    """
    ## Description

    Click a collapsible element to expand it.

    ## Example utterances

    * Expand the section below and click submit.

    ## Utterance fields

    (none)
    """

    subdomain = "click-collapsible"


class ClickCollapsible2Env(MiniWoBEnvironment):
    """
    ## Description

    Find and click on a specified link, from collapsible elements.

    ## Example utterances

    * Expand the sections below, to find and click on the link "Habitasse".
    * Expand the sections below, to find and click on the link "ac".
    * Expand the sections below, to find and click on the link "aliquet".
    * Expand the sections below, to find and click on the link "euismod.".
    * Expand the sections below, to find and click on the link "mattis".

    ## Utterance fields

    * target
    """

    subdomain = "click-collapsible-2"


class ClickCollapsible2NodelayEnv(MiniWoBEnvironment):
    """
    ## Description

    [click-collapsible-2] Removed animation.

    ## Example utterances

    * Expand the sections below, to find and click on the link "Habitasse".
    * Expand the sections below, to find and click on the link "ac".
    * Expand the sections below, to find and click on the link "aliquet".
    * Expand the sections below, to find and click on the link "euismod.".
    * Expand the sections below, to find and click on the link "mattis".

    ## Utterance fields

    * target
    """

    subdomain = "click-collapsible-2-nodelay"


class ClickCollapsibleNodelayEnv(MiniWoBEnvironment):
    """
    ## Description

    [click-collapsible] Removed animation.

    ## Example utterances

    * Expand the section below and click submit.

    ## Utterance fields

    (none)
    """

    subdomain = "click-collapsible-nodelay"


class ClickColorEnv(MiniWoBEnvironment):
    """
    ## Description

    Click the specified color.

    ## Example utterances

    * Click on the cyan colored box.
    * Click on the grey colored box.
    * Click on the olive colored box.
    * Click on the white colored box.
    * Click on the yellow colored box.

    ## Utterance fields

    * target

    ## Additional notes

    * The utterance will have the color name transcribed.
    """

    subdomain = "click-color"


class ClickDialogEnv(MiniWoBEnvironment):
    """
    ## Description

    Click the button to close the dialog box.

    ## Example utterances

    * Close the dialog box by clicking the "x".

    ## Utterance fields

    (none)
    """

    subdomain = "click-dialog"


class ClickDialog2Env(MiniWoBEnvironment):
    """
    ## Description

    Click a specific button in a dialog box.

    ## Example utterances

    * Click the button in the dialog box labeled "Cancel".
    * Click the button in the dialog box labeled "OK".
    * Click the button in the dialog box labeled "x".

    ## Utterance fields

    * target
    """

    subdomain = "click-dialog-2"


class ClickLinkEnv(MiniWoBEnvironment):
    """
    ## Description

    Click on a specified link in text.

    ## Example utterances

    * Click on the link "Eget".
    * Click on the link "blandit".
    * Click on the link "nam".
    * Click on the link "porttitor".
    * Click on the link "sed".

    ## Utterance fields

    * target
    """

    subdomain = "click-link"


class ClickMenuEnv(MiniWoBEnvironment):
    """
    ## Description

    Click menu items.

    ## Example utterances

    * Select Almira>Sherye>Anissa
    * Select Alvera>Terza>Ludovika
    * Select Diandra>Mollie>Velma
    * Select Evy>Tammara
    * Select Martelle

    ## Utterance fields

    * target
    """

    subdomain = "click-menu"


class ClickMenu2Env(MiniWoBEnvironment):
    """
    ## Description

    Find a specific item from a menu.

    ## Example utterances

    * Click the "Menu" button, and then find and click on the item labeled "Next".
    * Click the "Menu" button, and then find and click on the item labeled "Prev".
    * Click the "Menu" button, and then find and click on the item labeled "Save".
    * Click the "Menu" button, and then find and click on the item labeled "Zoom Out".
    * Click the "Menu" button, and then find and click on the item with the icon.

    ## Utterance fields

    * target
    """

    subdomain = "click-menu-2"


class ClickOptionEnv(MiniWoBEnvironment):
    """
    ## Description

    Click option boxes.

    ## Example utterances

    * Select AU and click Submit.
    * Select GDKkQ and click Submit.
    * Select Vdpn2dP and click Submit.
    * Select nZV6g2 and click Submit.
    * Select yS82i and click Submit.

    ## Utterance fields

    * target
    """

    subdomain = "click-option"


class ClickPieEnv(MiniWoBEnvironment):
    """
    ## Description

    Click items on a pie menu.

    ## Example utterances

    * Expand the pie menu below and click on the item labeled "7".
    * Expand the pie menu below and click on the item labeled "A".
    * Expand the pie menu below and click on the item labeled "S".
    * Expand the pie menu below and click on the item labeled "e".
    * Expand the pie menu below and click on the item labeled "f".

    ## Utterance fields

    * target
    """

    subdomain = "click-pie"


class ClickPieNodelayEnv(MiniWoBEnvironment):
    """
    ## Description

    [click-pie] Removed animation.

    ## Example utterances

    * Expand the pie menu below and click on the item labeled "7".
    * Expand the pie menu below and click on the item labeled "A".
    * Expand the pie menu below and click on the item labeled "S".
    * Expand the pie menu below and click on the item labeled "e".
    * Expand the pie menu below and click on the item labeled "f".

    ## Utterance fields

    * target
    """

    subdomain = "click-pie-nodelay"


class ClickScrollListEnv(MiniWoBEnvironment):
    """
    ## Description

    Click multiple items from a scroll list.

    ## Example utterances

    * Select Christal from the scroll list and click Submit.
    * Select Coleen from the scroll list and click Submit.
    * Select Corrine, Catherine from the scroll list and click Submit.
    * Select Heard Island and McDonald Islands, Nicaragua from the scroll list and click Submit.
    * Select Ronda from the scroll list and click Submit.

    ## Utterance fields

    * target
    """

    subdomain = "click-scroll-list"


class ClickShadesEnv(MiniWoBEnvironment):
    """
    ## Description

    Click the shades that match a specified color.

    ## Example utterances

    * Select all the shades of blue and press Submit.
    * Select all the shades of green and press Submit.
    * Select all the shades of red and press Submit.

    ## Utterance fields

    * target
    """

    subdomain = "click-shades"


class ClickShapeEnv(MiniWoBEnvironment):
    """
    ## Description

    Click on a specific shape.

    ## Example utterances

    * Click on a 9
    * Click on a aqua item
    * Click on a large digit
    * Click on a small green shape
    * Click on a small letter

    ## Utterance fields

    * color
    * size
    * target
    * type
    """

    subdomain = "click-shape"


class ClickTabEnv(MiniWoBEnvironment):
    """
    ## Description

    Click on a tab element.

    ## Example utterances

    * Click on Tab #1.
    * Click on Tab #2.
    * Click on Tab #3.

    ## Utterance fields

    * target
    """

    subdomain = "click-tab"


class ClickTab2Env(MiniWoBEnvironment):
    """
    ## Description

    Click a link inside a specific tab element.

    ## Example utterances

    * Switch between the tabs to find and click on the link "Habitasse".
    * Switch between the tabs to find and click on the link "ac".
    * Switch between the tabs to find and click on the link "aliquet".
    * Switch between the tabs to find and click on the link "euismod.".
    * Switch between the tabs to find and click on the link "mattis".

    ## Utterance fields

    * target
    """

    subdomain = "click-tab-2"


class ClickTab2EasyEnv(MiniWoBEnvironment):
    """
    ## Description

    [click-tab-2] One 1 tab.

    ## Example utterances

    * Switch between the tabs to find and click on the link "Eget".
    * Switch between the tabs to find and click on the link "blandit".
    * Switch between the tabs to find and click on the link "nam".
    * Switch between the tabs to find and click on the link "porttitor".
    * Switch between the tabs to find and click on the link "sed".

    ## Utterance fields

    * target
    """

    subdomain = "click-tab-2-easy"


class ClickTab2HardEnv(MiniWoBEnvironment):
    """
    ## Description

    [click-tab-2] Varying number of tabs from 2 to 6.

    ## Example utterances

    * Switch between the tabs to find and click on the link "Duis".
    * Switch between the tabs to find and click on the link "Sed".
    * Switch between the tabs to find and click on the link "Viverra".
    * Switch between the tabs to find and click on the link "lectus".
    * Switch between the tabs to find and click on the link "vulputate.".

    ## Utterance fields

    * target
    """

    subdomain = "click-tab-2-hard"


class ClickTab2MediumEnv(MiniWoBEnvironment):
    """
    ## Description

    [click-tab-2] Choose between a link or ‘no match’.

    ## Example utterances

    * Switch between the tabs to find and click on the link "Congue.".
    * Switch between the tabs to find and click on the link "Nisl".
    * Switch between the tabs to find and click on the link "aliquet".
    * Switch between the tabs to find and click on the link "justo.".
    * Switch between the tabs to find and click on the link "mattis".

    ## Utterance fields

    * target

    ## Additional notes

    * The second tab should only be clicked if the specified link is not in the current tab.
    """

    subdomain = "click-tab-2-medium"


class ClickTestEnv(MiniWoBEnvironment):
    """
    ## Description

    Click on a single button.

    ## Example utterances

    * Click the button.

    ## Utterance fields

    (none)
    """

    subdomain = "click-test"


class ClickTest2Env(MiniWoBEnvironment):
    """
    ## Description

    Click on one of two buttons.

    ## Example utterances

    * Click button ONE.

    ## Utterance fields

    * target
    """

    subdomain = "click-test-2"


class ClickTestTransferEnv(MiniWoBEnvironment):
    """
    ## Description

    [click-test] Different buttons during train and test.

    ## Example utterances

    * Click button ONE.

    ## Utterance fields

    * target

    ## Additional notes

    * Use the `set_data_mode` method of the environment to switch between the train and test scenarios.
    """

    subdomain = "click-test-transfer"


class ClickWidgetEnv(MiniWoBEnvironment):
    """
    ## Description

    Click on a specific widget in a generated form.

    ## Example utterances

    * Click on a "button" widget.
    * Click on a "checkbox" widget.
    * Click on a "radio" widget.
    * Click on a "text" widget.
    * Click on a "textarea" widget.

    ## Utterance fields

    * target
    """

    subdomain = "click-widget"


class CopyPasteEnv(MiniWoBEnvironment):
    """
    ## Description

    Copy text and paste it into an input.

    ## Example utterances

    * Copy the text in the textarea below, paste it into the textbox and press Submit.

    ## Utterance fields

    (none)

    ## Additional notes

    * The input is checked against the original content of the source textarea.
    """

    subdomain = "copy-paste"


class CopyPaste2Env(MiniWoBEnvironment):
    """
    ## Description

    Copy text from a specific textarea and paste it into an input.

    ## Example utterances

    * Copy the text from the 1st text area below and paste it into the text input, then press Submit.
    * Copy the text from the 2nd text area below and paste it into the text input, then press Submit.
    * Copy the text from the 3rd text area below and paste it into the text input, then press Submit.

    ## Utterance fields

    * target

    ## Additional notes

    * The input is checked against the original content of the source textarea.
    """

    subdomain = "copy-paste-2"


class CountShapeEnv(MiniWoBEnvironment):
    """
    ## Description

    Count number of shapes.

    ## Example utterances

    * How many aqua items are there?
    * How many large yellow items are there?
    * How many red digits are there?
    * How many shapes are there?
    * How many small green items are there?

    ## Utterance fields

    * color
    * size
    * target
    * type
    """

    subdomain = "count-shape"


class CountSidesEnv(MiniWoBEnvironment):
    """
    ## Description

    Count the number of sides on a shape.

    ## Example utterances

    * Press the button that correctly denotes how many sides the shape has.

    ## Utterance fields

    (none)
    """

    subdomain = "count-sides"


class DailyCalendarEnv(MiniWoBEnvironment):
    """
    ## Description

    Create an event on a daily calendar.

    ## Example utterances

    * Create a 0.5 hours event named "Food", between 8AM and 12PM.
    * Create a 1.5 hours event named "Food", between 4PM and 8PM.
    * Create a 30 mins event named "Meeting", between 4PM and 8PM.
    * Create a 60 mins event named "Meeting", between 4PM and 8PM.
    * Create a 90 mins event named "Phonecall", between 12PM and 4PM.

    ## Utterance fields

    * between from
    * between to
    * length
    * name

    ## Additional notes

    * The created event must not overlap with existing events.
    """

    subdomain = "daily-calendar"


class DragBoxEnv(MiniWoBEnvironment):
    """
    ## Description

    Drag the smaller box into the larger box.

    ## Example utterances

    * Drag the smaller box so that it is completely inside the larger box.

    ## Utterance fields

    (none)
    """

    subdomain = "drag-box"


class DragCircleEnv(MiniWoBEnvironment):
    """
    ## Description

    Drag an item in a specified direction.

    ## Example utterances

    * Drag the circle down then press Submit.
    * Drag the circle left then press Submit.
    * Drag the circle right then press Submit.
    * Drag the circle up then press Submit.

    ## Utterance fields

    * target
    """

    subdomain = "drag-circle"


class DragCubeEnv(MiniWoBEnvironment):
    """
    ## Description

    Drag a 3D cube to show a specific face.

    ## Example utterances

    * Move the cube around so that "1" is the active side facing the user.
    * Move the cube around so that "2" is the active side facing the user.
    * Move the cube around so that "3" is the active side facing the user.
    * Move the cube around so that "5" is the active side facing the user.
    * Move the cube around so that "6" is the active side facing the user.

    ## Utterance fields

    * target

    ## Additional notes

    * Non-deterministic: How much the cube moves depends on the mouse speed in real time.
    """

    subdomain = "drag-cube"


class DragItemsEnv(MiniWoBEnvironment):
    """
    ## Description

    Drag items in a list, in a specified direction

    ## Example utterances

    * Drag Gerda to the 3rd position.
    * Drag Jerrilyn to the 1st position.
    * Drag Margette down by one position.
    * Drag Sara-Ann to the 1st position.
    * Drag Theodosia to the 3rd position.

    ## Utterance fields

    * target
    """

    subdomain = "drag-items"


class DragItemsGridEnv(MiniWoBEnvironment):
    """
    ## Description

    Drag items in a 2D grid around.

    ## Example utterances

    * Drag Andria to the top right.
    * Drag Drucy up by one.
    * Drag Marcille to the bottom right.
    * Drag Marta to the bottom center.
    * Drag Vally to the bottom center.

    ## Utterance fields

    * target
    """

    subdomain = "drag-items-grid"


class DragShapesEnv(MiniWoBEnvironment):
    """
    ## Description

    Drag shapes into a box.

    ## Example utterances

    * Drag all circles into the black box.
    * Drag all rectangles into the black box.
    * Drag all triangles into the black box.

    ## Utterance fields

    * target
    """

    subdomain = "drag-shapes"


class DragShapes2Env(MiniWoBEnvironment):
    """
    ## Description

    Drag shapes into boxes, categorized by type.

    ## Example utterances

    * Drag all circles into the left box, and everything else into the right box.
    * Drag all magenta shapes into the left box, and everything else into the right box.
    * Drag all rectangles into the left box, and everything else into the right box.
    * Drag all red shapes into the left box, and everything else into the right box.
    * Drag all yellow shapes into the left box, and everything else into the right box.

    ## Utterance fields

    * target

    ## Additional notes

    * **Partial reward:** The reward is (1.3 x % correct) - (% incorrect), clipped to range -1 to 1.
    """

    subdomain = "drag-shapes-2"


class DragSingleShapeEnv(MiniWoBEnvironment):
    """
    ## Description

    Drag a randomly generated shape in a specified direction.

    ## Example utterances

    * Drag the item down then press Submit.
    * Drag the item left then press Submit.
    * Drag the item right then press Submit.
    * Drag the item up then press Submit.

    ## Utterance fields

    * target
    """

    subdomain = "drag-single-shape"


class DragSortNumbersEnv(MiniWoBEnvironment):
    """
    ## Description

    Drag numbers into sorted ascending order.

    ## Example utterances

    * Sort the numbers in increasing order, starting with the lowest number at the top of the list.

    ## Utterance fields

    (none)
    """

    subdomain = "drag-sort-numbers"


class DrawCircleEnv(MiniWoBEnvironment):
    """
    ## Description

    Draw a circle around a marked point.

    ## Example utterances

    * Draw a circle centered around the marked point by dragging the mouse. Press submit when done.

    ## Utterance fields

    (none)

    ## Additional notes

    * **Partial reward:** The reward is -0.25 if the shape is too small. Otherwise the reward is based on how big the shape is and how consistent the circle radius is.
    """

    subdomain = "draw-circle"


class DrawLineEnv(MiniWoBEnvironment):
    """
    ## Description

    Draw a line through a marked point.

    ## Example utterances

    * Draw a horizontal line that runs through the dot, then press submit.
    * Draw a vertical line that runs through the dot, then press submit.

    ## Utterance fields

    * direction

    ## Additional notes

    * **Partial reward:** The reward is based on the line direction and the distance from the marked point.
    """

    subdomain = "draw-line"


class EmailInboxEnv(MiniWoBEnvironment):
    """
    ## Description

    Navigate through an email inbox and perform some actions.

    ## Example utterances

    * Find the email by Audrey and forward that email to Hedy.
    * Find the email by Bettine and reply to them with the text "Gravida.".
    * Find the email by Cathrine and click the star icon to mark it as important.
    * Find the email by Jemima and reply to them with the text "Purus velit faucibus eu.".
    * Find the email by Valida and reply to them with the text "Risus augue odio turpis.".

    ## Utterance fields

    * by
    * message
    * task
    * to
    """

    subdomain = "email-inbox"


class EmailInboxDeleteEnv(MiniWoBEnvironment):
    """
    ## Description

    [email-inbox] No scrolling + 1 subtask.

    ## Example utterances

    * Find the email by Audrey and click the trash icon to delete it.
    * Find the email by Coletta and click the trash icon to delete it.
    * Find the email by Danice and click the trash icon to delete it.
    * Find the email by Gerta and click the trash icon to delete it.
    * Find the email by Goldina and click the trash icon to delete it.

    ## Utterance fields

    * by
    * task
    """

    subdomain = "email-inbox-delete"


class EmailInboxForwardEnv(MiniWoBEnvironment):
    """
    ## Description

    [email-inbox] No scrolling + 1 subtask.

    ## Example utterances

    * Find the email by Audrey and forward that email to Orel.
    * Find the email by Coletta and forward that email to Odette.
    * Find the email by Danice and forward that email to Camel.
    * Find the email by Gerta and forward that email to Marna.
    * Find the email by Goldina and forward that email to Judith.

    ## Utterance fields

    * by
    * task
    * to
    """

    subdomain = "email-inbox-forward"


class EmailInboxForwardNlEnv(MiniWoBEnvironment):
    """
    ## Description

    [email-inbox-forward] varied instruction texts (30 templates).

    ## Example utterances

    * Amargo is waiting for the email by Lissie.
    * Forward to Cathrine the email from Evy.
    * Give Bobine the message you received from Cora,
    * Please forward to Delores the email by Valida.
    * Send Bettine the information Lidia sent to you.

    ## Utterance fields

    * by
    * to

    ## Additional notes

    * Test instruction templates are different from training one. Use the `set_data_mode` method of the environment to switch between the train and test templates.
    """

    subdomain = "email-inbox-forward-nl"


class EmailInboxForwardNlTurkEnv(MiniWoBEnvironment):
    """
    ## Description

    [email-inbox-forward] varied instruction texts (100 templates).

    ## Example utterances

    * Amargo wrote an email in this app that needs to be forwarded to Penelope.
    * Find Coletta's email and send it to Evy.
    * Look for the email that Bobine wrote in this email app and forward it to Bevvy
    * Open Patsy's email and forward it to Delores
    * Will you find the email from Kiley and forward it to Sibylla?

    ## Utterance fields

    * by
    * to

    ## Additional notes

    * Test instruction templates are different from training one. Use the `set_data_mode` method of the environment to switch between the train and test templates.
    """

    subdomain = "email-inbox-forward-nl-turk"


class EmailInboxImportantEnv(MiniWoBEnvironment):
    """
    ## Description

    [email-inbox] No scrolling + 1 subtask.

    ## Example utterances

    * Find the email by Audrey and click the star icon to mark it as important.
    * Find the email by Coletta and click the star icon to mark it as important.
    * Find the email by Danice and click the star icon to mark it as important.
    * Find the email by Gerta and click the star icon to mark it as important.
    * Find the email by Goldina and click the star icon to mark it as important.

    ## Utterance fields

    * by
    * task
    """

    subdomain = "email-inbox-important"


class EmailInboxNlTurkEnv(MiniWoBEnvironment):
    """
    ## Description

    [email-inbox] varied instruction texts (100 templates for each subtask).

    ## Example utterances

    * Bobine's email should be deleted from the inbox.
    * Delete all messages from Coletta.
    * Email Kiley the message, "Amet, a. Malesuada. Nunc."
    * Please delete Amargo's emails.
    * Please find the email Patsy sent me and file it as important

    ## Utterance fields

    * by
    * message
    * task
    * to

    ## Additional notes

    * Test instruction templates are different from training one. Use the `set_data_mode` method of the environment to switch between the train and test templates.
    """

    subdomain = "email-inbox-nl-turk"


class EmailInboxNoscrollEnv(MiniWoBEnvironment):
    """
    ## Description

    [email-inbox] No scrolling.

    ## Example utterances

    * Find the email by Audrey and forward that email to Orel.
    * Find the email by Coletta and reply to them with the text "Odio fermentum quam auctor.".
    * Find the email by Danice and click the trash icon to delete it.
    * Find the email by Gerta and click the trash icon to delete it.
    * Find the email by Goldina and click the star icon to mark it as important.

    ## Utterance fields

    * by
    * message
    * task
    * to
    """

    subdomain = "email-inbox-noscroll"


class EmailInboxReplyEnv(MiniWoBEnvironment):
    """
    ## Description

    [email-inbox] No scrolling + 1 subtask.

    ## Example utterances

    * Find the email by Audrey and reply to them with the text "Vel quis. Semper.".
    * Find the email by Coletta and reply to them with the text "Odio fermentum quam auctor.".
    * Find the email by Danice and reply to them with the text "Aenean.".
    * Find the email by Gerta and reply to them with the text "Ac sit non.".
    * Find the email by Goldina and reply to them with the text "At non lacus.".

    ## Utterance fields

    * by
    * message
    * task
    """

    subdomain = "email-inbox-reply"


class EmailInboxStarReplyEnv(MiniWoBEnvironment):
    """
    ## Description

    [email-inbox] No scrolling + 2 subtasks.

    ## Example utterances

    * Find the email by Audrey and reply to them with the text "Vel quis. Semper.".
    * Find the email by Coletta and reply to them with the text "Odio fermentum quam auctor.".
    * Find the email by Danice and click the star icon to mark it as important.
    * Find the email by Gerta and click the star icon to mark it as important.
    * Find the email by Goldina and click the star icon to mark it as important.

    ## Utterance fields

    * by
    * message
    * task
    """

    subdomain = "email-inbox-star-reply"


class EnterDateEnv(MiniWoBEnvironment):
    """
    ## Description

    Use the date input to pick the correct date.

    ## Example utterances

    * Enter 05/20/2010 as the date and hit submit.
    * Enter 07/26/2017 as the date and hit submit.
    * Enter 09/17/2013 as the date and hit submit.
    * Enter 10/02/2013 as the date and hit submit.
    * Enter 10/11/2011 as the date and hit submit.

    ## Utterance fields

    * target
    """

    subdomain = "enter-date"


class EnterPasswordEnv(MiniWoBEnvironment):
    """
    ## Description

    Enter the password into the form.

    ## Example utterances

    * Enter the password "817" into both text fields and press submit.
    * Enter the password "D91YP" into both text fields and press submit.
    * Enter the password "Sfz" into both text fields and press submit.
    * Enter the password "fU" into both text fields and press submit.
    * Enter the password "yA" into both text fields and press submit.

    ## Utterance fields

    * target
    """

    subdomain = "enter-password"


class EnterTextEnv(MiniWoBEnvironment):
    """
    ## Description

    Enter given text to a textfield.

    ## Example utterances

    * Enter "Agustina" into the text field and press Submit.
    * Enter "Ignacio" into the text field and press Submit.
    * Enter "Jerald" into the text field and press Submit.
    * Enter "Marcella" into the text field and press Submit.
    * Enter "Myron" into the text field and press Submit.

    ## Utterance fields

    * target
    """

    subdomain = "enter-text"


class EnterText2Env(MiniWoBEnvironment):
    """
    ## Description

    Convert given text to upper or lower case.

    ## Example utterances

    * Type "IGNACIO" in all lower case letters in the text input and press Submit.
    * Type "JERALD" in all lower case letters in the text input and press Submit.
    * Type "MARCELLA" in all lower case letters in the text input and press Submit.
    * Type "agustina" in all upper case letters in the text input and press Submit.
    * Type "myron" in all upper case letters in the text input and press Submit.

    ## Utterance fields

    * case
    * text
    """

    subdomain = "enter-text-2"


class EnterTextDynamicEnv(MiniWoBEnvironment):
    """
    ## Description

    Enter dynamically generated text to a textfield.

    ## Example utterances

    * Enter "6gH" into the text field and press Submit.
    * Enter "YPFe2" into the text field and press Submit.
    * Enter "aIGD" into the text field and press Submit.
    * Enter "jRPft" into the text field and press Submit.
    * Enter "zq" into the text field and press Submit.

    ## Utterance fields

    * target
    """

    subdomain = "enter-text-dynamic"


class EnterTimeEnv(MiniWoBEnvironment):
    """
    ## Description

    Enter the specified time into the input.

    ## Example utterances

    * Enter 12:25 AM as the time and press submit.
    * Enter 12:34 PM as the time and press submit.
    * Enter 7:01 AM as the time and press submit.
    * Enter 8:04 PM as the time and press submit.
    * Enter 9:37 PM as the time and press submit.

    ## Utterance fields

    * target
    """

    subdomain = "enter-time"


class FindGreatestEnv(MiniWoBEnvironment):
    """
    ## Description

    Find the card with the greatest number.

    ## Example utterances

    * Find and pick the card with the greatest number, then press submit.

    ## Utterance fields

    (none)
    """

    subdomain = "find-greatest"


class FindMidpointEnv(MiniWoBEnvironment):
    """
    ## Description

    Find the shortest mid-point of two points.

    ## Example utterances

    * Find and click on the shortest mid-point between the two points, then press submit.

    ## Utterance fields

    (none)

    ## Additional notes

    * **Partial reward:** If the point is at most 30px away from the midpoint, the partial reward is how close it is to the midpoint (from 0 to 1).
    * Since the reward is a continuous value, use the `get_thresholded_reward` reward preprocessor to binarize the reward instead of `get_binary_reward`.
    """

    subdomain = "find-midpoint"


class FindWordEnv(MiniWoBEnvironment):
    """
    ## Description

    Find nth word in a block of text.

    ## Example utterances

    * Find the 1st word in the paragraph, type that into the textbox and press "Submit".
    * Find the 2nd word in the paragraph, type that into the textbox and press "Submit".
    * Find the 6th word in the paragraph, type that into the textbox and press "Submit".
    * Find the 8th word in the paragraph, type that into the textbox and press "Submit".
    * Find the 9th word in the paragraph, type that into the textbox and press "Submit".

    ## Utterance fields

    * target
    """

    subdomain = "find-word"


class FocusTextEnv(MiniWoBEnvironment):
    """
    ## Description

    Focus into a text input.

    ## Example utterances

    * Focus into the textbox.

    ## Utterance fields

    (none)
    """

    subdomain = "focus-text"


class FocusText2Env(MiniWoBEnvironment):
    """
    ## Description

    Focus on a specific text input.

    ## Example utterances

    * Focus into the 1st input textbox.
    * Focus into the 2nd input textbox.
    * Focus into the 3rd input textbox.

    ## Utterance fields

    * target
    """

    subdomain = "focus-text-2"


class FormSequenceEnv(MiniWoBEnvironment):
    """
    ## Description

    Perform a series of instructions on a form.

    ## Example utterances

    * Select -4 with the slider, click the 1st checkbox, then hit Submit.
    * Select 1 with the slider, click the 3rd checkbox, then hit Submit.
    * Select 2 with the slider, click the 1st checkbox, then hit Submit.
    * Select 7 with the slider, click the 2nd checkbox, then hit Submit.
    * Select 9 with the slider, click the 1st checkbox, then hit Submit.

    ## Utterance fields

    * checkbox target
    * slider target
    """

    subdomain = "form-sequence"


class FormSequence2Env(MiniWoBEnvironment):
    """
    ## Description

    Perform a series of instructions on a form.

    ## Example utterances

    * Check the 1st radio button and enter the number "-5" into the 2nd textbox.
    * Check the 1st radio button and enter the number "3" into the 2nd textbox.
    * Check the 2nd radio button and enter the number "26" into the 1st textbox.
    * Check the 2nd radio button and enter the number "48" into the 3rd textbox.
    * Check the 3rd radio button and enter the number "27" into the 3rd textbox.

    ## Utterance fields

    * radio target
    * textbox target
    * textbox text
    """

    subdomain = "form-sequence-2"


class FormSequence3Env(MiniWoBEnvironment):
    """
    ## Description

    Perform a series of instructions on a form.

    ## Example utterances

    * Choose 5ft 10in from the dropdown, then click the button labeled "Maybe".
    * Choose 5ft 10in from the dropdown, then click the button labeled "No".
    * Choose 5ft 9in from the dropdown, then click the button labeled "Yes".
    * Choose 6 ft from the dropdown, then click the button labeled "No".
    * Choose 6ft 1in from the dropdown, then click the button labeled "Yes".

    ## Utterance fields

    * button target
    * dropdown target
    """

    subdomain = "form-sequence-3"


class GenerateNumberEnv(MiniWoBEnvironment):
    """
    ## Description

    Generate a random number that meets certain criteria.

    ## Example utterances

    * Generate a number greater than 4, then press submit.
    * Generate a number greater than 5, then press submit.
    * Generate a number less than 5, then press submit.
    * Generate a number less than 8, then press submit.
    * Generate an even number, then press submit.

    ## Utterance fields

    * criterion
    * number
    """

    subdomain = "generate-number"


class GridCoordinateEnv(MiniWoBEnvironment):
    """
    ## Description

    Find the Cartesian coordinates on a grid.

    ## Example utterances

    * Click on the grid coordinate (-1,1).
    * Click on the grid coordinate (-1,2).
    * Click on the grid coordinate (-2,-1).
    * Click on the grid coordinate (-2,-2).
    * Click on the grid coordinate (1,1).

    ## Utterance fields

    * x
    * y
    """

    subdomain = "grid-coordinate"


class GuessNumberEnv(MiniWoBEnvironment):
    """
    ## Description

    Guess the number.

    ## Example utterances

    * Guess the number between 0-9 and press Submit. Use the feedback below to find the right number.

    ## Utterance fields

    (none)

    ## Additional notes

    * Can make as many guesses as needed.
    """

    subdomain = "guess-number"


class HighlightTextEnv(MiniWoBEnvironment):
    """
    ## Description

    Highlight all the text.

    ## Example utterances

    * Highlight the text in the paragraph below and click submit.

    ## Utterance fields

    (none)
    """

    subdomain = "highlight-text"


class HighlightText2Env(MiniWoBEnvironment):
    """
    ## Description

    Highlight the specified paragraph.

    ## Example utterances

    * Highlight the text in the 1st paragraph and click submit.
    * Highlight the text in the 2nd paragraph and click submit.
    * Highlight the text in the 3rd paragraph and click submit.

    ## Utterance fields

    * target
    """

    subdomain = "highlight-text-2"


class HotColdEnv(MiniWoBEnvironment):
    """
    ## Description

    Find and click on the hot area.

    ## Example utterances

    * Find and click on the HOT area.

    ## Utterance fields

    * target

    ## Additional notes

    * **Partial reward:** The partial reward is 0.5 for clicking a "warm" area, and 0.25 for a "cold" area.
    """

    subdomain = "hot-cold"


class IdentifyShapeEnv(MiniWoBEnvironment):
    """
    ## Description

    Identify a randomly generated shape.

    ## Example utterances

    * Click the button that best describes the figure below.

    ## Utterance fields

    (none)
    """

    subdomain = "identify-shape"


class LoginUserEnv(MiniWoBEnvironment):
    """
    ## Description

    Enter user login details into the form.

    ## Example utterances

    * Enter the username "karrie" and the password "AU" into the text fields and press login.
    * Enter the username "keneth" and the password "91YP" into the text fields and press login.
    * Enter the username "nathalie" and the password "17jRP" into the text fields and press login.
    * Enter the username "nathalie" and the password "fzzq" into the text fields and press login.
    * Enter the username "vina" and the password "US" into the text fields and press login.

    ## Utterance fields

    * password
    * username
    """

    subdomain = "login-user"


class LoginUserPopupEnv(MiniWoBEnvironment):
    """
    ## Description

    [login-user] Random popup.

    ## Example utterances

    * Enter the username "karrie" and the password "AU" into the text fields and press login.
    * Enter the username "keneth" and the password "91YP" into the text fields and press login.
    * Enter the username "nathalie" and the password "17jRP" into the text fields and press login.
    * Enter the username "nathalie" and the password "fzzq" into the text fields and press login.
    * Enter the username "vina" and the password "US" into the text fields and press login.

    ## Utterance fields

    * password
    * username
    """

    subdomain = "login-user-popup"


class MultiLayoutsEnv(MiniWoBEnvironment):
    """
    ## Description

    Fill in forms of varying layouts.

    ## Example utterances

    * Search for action movies directed by Bowen from year 1998.
    * Search for crime movies directed by Curry from year 1995.
    * Search for historical movies directed by Mcknight from year 1980.
    * Search for historical movies directed by West from year 2010.
    * Search for satire movies directed by Mejia from year 2017.

    ## Utterance fields

    * director
    * genre
    * year
    """

    subdomain = "multi-layouts"


class MultiOrderingsEnv(MiniWoBEnvironment):
    """
    ## Description

    Fill in forms with shuffled field orderings.

    ## Example utterances

    * Search for action movies directed by Bowen from year 1998.
    * Search for crime movies directed by Curry from year 1995.
    * Search for historical movies directed by Mcknight from year 1980.
    * Search for historical movies directed by West from year 2010.
    * Search for satire movies directed by Mejia from year 2017.

    ## Utterance fields

    * director
    * genre
    * year
    """

    subdomain = "multi-orderings"


class NavigateTreeEnv(MiniWoBEnvironment):
    """
    ## Description

    Navigate a file tree to find a specified file or folder.

    ## Example utterances

    * Navigate through the file tree. Find and click on the folder or file named "Agustina".
    * Navigate through the file tree. Find and click on the folder or file named "Annis".
    * Navigate through the file tree. Find and click on the folder or file named "Briana".
    * Navigate through the file tree. Find and click on the folder or file named "Deneen".
    * Navigate through the file tree. Find and click on the folder or file named "Truman".

    ## Utterance fields

    * target
    """

    subdomain = "navigate-tree"


class NumberCheckboxesEnv(MiniWoBEnvironment):
    """
    ## Description

    Draw a given number using checkboxes.

    ## Example utterances

    * Draw the number "0" in the checkboxes using the example on the right and press Submit when finished.
    * Draw the number "1" in the checkboxes using the example on the right and press Submit when finished.
    * Draw the number "3" in the checkboxes using the example on the right and press Submit when finished.
    * Draw the number "7" in the checkboxes using the example on the right and press Submit when finished.
    * Draw the number "8" in the checkboxes using the example on the right and press Submit when finished.

    ## Utterance fields

    * target
    """

    subdomain = "number-checkboxes"


class OddOrEvenEnv(MiniWoBEnvironment):
    """
    ## Description

    Mark each number as odd or even.

    ## Example utterances

    * Mark the numbers below as odd or even and press submit when done.

    ## Utterance fields

    (none)

    ## Additional notes

    * **Partial reward:** The score for each number is 1 for correct, -1 for incorrect, and -0.25 for unselected. The final reward is the average score.
    """

    subdomain = "odd-or-even"


class OrderFoodEnv(MiniWoBEnvironment):
    """
    ## Description

    Order food items from a menu.

    ## Example utterances

    * Order 3 items that are vegan
    * Order one of each item: Coconut mango tart, Ice cream sundae
    * Order one of each item: Garlic bread, Spaghetti and Meatballs
    * Order one of each item: Spaghetti and Meatballs, Spinach and goat cheese dip
    * Order one of each item: Spinach and goat cheese dip, Grilled Pork Tenderloin

    ## Utterance fields

    * amount
    * criterion
    * target 0
    * target 1
    * type
    """

    subdomain = "order-food"


class PhoneBookEnv(MiniWoBEnvironment):
    """
    ## Description

    Find a contact in a phone book.

    ## Example utterances

    * Find Casie in the contact book and click on their address.
    * Find Eadith in the contact book and click on their email.
    * Find Sayre in the contact book and click on their address.
    * Find Verile in the contact book and click on their phone number.
    * Find Wilow in the contact book and click on their address.

    ## Utterance fields

    * name
    * target

    ## Additional notes

    * **Partial reward:** The partial reward is 0.7 for the correct person and 0.3 for the correct contact type.
    """

    subdomain = "phone-book"


class ReadTableEnv(MiniWoBEnvironment):
    """
    ## Description

    Read information out from a table.

    ## Example utterances

    * Enter the value of Color into the text field and press Submit.
    * Enter the value of Country into the text field and press Submit.
    * Enter the value of Language into the text field and press Submit.
    * Enter the value of Last name into the text field and press Submit.
    * Enter the value of Religion into the text field and press Submit.

    ## Utterance fields

    * target
    """

    subdomain = "read-table"


class ReadTable2Env(MiniWoBEnvironment):
    """
    ## Description

    Read multiple pieces of information out from a table.

    ## Example utterances

    * Enter the value that corresponds with each label into the form and submit when done.

    ## Utterance fields

    (none)
    """

    subdomain = "read-table-2"


class ResizeTextareaEnv(MiniWoBEnvironment):
    """
    ## Description

    Resize a textarea in a given direction.

    ## Example utterances

    * Resize the textarea so that the height is larger than its initial size then press Submit.
    * Resize the textarea so that the height is smaller than its initial size then press Submit.
    * Resize the textarea so that the width is larger than its initial size then press Submit.
    * Resize the textarea so that the width is smaller than its initial size then press Submit.

    ## Utterance fields

    * target
    """

    subdomain = "resize-textarea"


class RightAngleEnv(MiniWoBEnvironment):
    """
    ## Description

    Given two points, add a third point to create a right angle.

    ## Example utterances

    * Add a third point to create a right angle, then press submit.

    ## Utterance fields

    (none)

    ## Additional notes

    * **Partial reward:** If the angle is less than 45 degrees away from a right angle, the partial reward is how close it is to a right angle (from 0 to 1).
    """

    subdomain = "right-angle"


class ScrollTextEnv(MiniWoBEnvironment):
    """
    ## Description

    Scroll through a text area element and enter last word into text area.

    ## Example utterances

    * Find the last word in the text area, enter it into the text field and hit Submit.

    ## Utterance fields

    (none)
    """

    subdomain = "scroll-text"


class ScrollText2Env(MiniWoBEnvironment):
    """
    ## Description

    Scroll through a text area in a given direction.

    ## Example utterances

    * Scroll the textarea to the bottom of the text hit submit.
    * Scroll the textarea to the top of the text hit submit.

    ## Utterance fields

    * target
    """

    subdomain = "scroll-text-2"


class SearchEngineEnv(MiniWoBEnvironment):
    """
    ## Description

    Search through a bunch of results to find a specified link.

    ## Example utterances

    * Use the textbox to enter "Dolores" and press "Search", then find and click the 7th search result.
    * Use the textbox to enter "Kanesha" and press "Search", then find and click the 4th search result.
    * Use the textbox to enter "Nathalie" and press "Search", then find and click the 1st search result.
    * Use the textbox to enter "Renda" and press "Search", then find and click the 7th search result.
    * Use the textbox to enter "Riley" and press "Search", then find and click the 7th search result.

    ## Utterance fields

    * query
    * rank
    """

    subdomain = "search-engine"


class SignAgreementEnv(MiniWoBEnvironment):
    """
    ## Description

    Sign a user agreement.

    ## Example utterances

    * Click the cancel button.
    * Scroll to the bottom of the textarea, enter the name "Cristin" then press "Cancel"
    * Scroll to the bottom of the textarea, enter the name "Juan" then press "Cancel"
    * Scroll to the bottom of the textarea, enter the name "Olin" then press "Agree"
    * Scroll to the bottom of the textarea, enter the name "Vanda" then press "Agree"

    ## Utterance fields

    * button
    * name
    """

    subdomain = "sign-agreement"


class SimpleAlgebraEnv(MiniWoBEnvironment):
    """
    ## Description

    Solve for X.

    ## Example utterances

    * Solve for x and type your answer into the textbox. Press Submit when done.

    ## Utterance fields

    (none)
    """

    subdomain = "simple-algebra"


class SimpleArithmeticEnv(MiniWoBEnvironment):
    """
    ## Description

    Perform some arithmetic math operations.

    ## Example utterances

    * Solve the math problem and type your answer into the textbox. Press submit when done.

    ## Utterance fields

    (none)
    """

    subdomain = "simple-arithmetic"


class SocialMediaEnv(MiniWoBEnvironment):
    """
    ## Description

    Interact with a social media feed.

    ## Example utterances

    * For the user @consectetur, click on the "Block" button.
    * For the user @emile, click on the "Embed Tweet" button.
    * For the user @leo, click on the "Mute" button.
    * For the user @leo, click on the "Share via DM" button.
    * For the user @nec, click on the "Share via DM" button.

    ## Utterance fields

    * button
    * user
    """

    subdomain = "social-media"


class SocialMediaAllEnv(MiniWoBEnvironment):
    """
    ## Description

    [social-media] Do some action on all matching entries.

    ## Example utterances

    * Click the "Like" button on all posts by @deneen and then click Submit.
    * Click the "Reply" button on all posts by @dis and then click Submit.
    * Click the "Retweet" button on all posts by @leonie and then click Submit.
    * Click the "Share" button on all posts by @annis and then click Submit.
    * Click the "Share" button on all posts by @egestas and then click Submit.

    ## Utterance fields

    * button
    * submit
    * user
    """

    subdomain = "social-media-all"


class SocialMediaSomeEnv(MiniWoBEnvironment):
    """
    ## Description

    [social-media] Do some action on some matching entries.

    ## Example utterances

    * Click the "Like" button on 5 posts by @deneen and then click Submit.
    * Click the "Reply" button on 2 posts by @dis and then click Submit.
    * Click the "Retweet" button on 3 posts by @leonie and then click Submit.
    * Click the "Share" button on 1 post by @annis and then click Submit.
    * Click the "Share" button on 8 posts by @egestas and then click Submit.

    ## Utterance fields

    * amount
    * button
    * submit
    * user
    """

    subdomain = "social-media-some"


class StockMarketEnv(MiniWoBEnvironment):
    """
    ## Description

    Buy from the stock market below a specified price.

    ## Example utterances

    * Buy ACP stock when the price is less than $56.60.
    * Buy EFN stock when the price is less than $49.60.
    * Buy JPF stock when the price is less than $45.20.
    * Buy JYV stock when the price is less than $62.10.
    * Buy TPZ stock when the price is less than $48.20.

    ## Utterance fields

    * name
    * target
    """

    subdomain = "stock-market"


class TerminalEnv(MiniWoBEnvironment):
    """
    ## Description

    Use the terminal to delete a file.

    ## Example utterances

    * Use the terminal below to delete a file ending with the extension .gif
    * Use the terminal below to delete a file ending with the extension .gpg
    * Use the terminal below to delete a file ending with the extension .rb
    * Use the terminal below to delete a file ending with the extension .tar.gz
    * Use the terminal below to delete a file that has no file extension.

    ## Utterance fields

    * target
    """

    subdomain = "terminal"


class TextEditorEnv(MiniWoBEnvironment):
    """
    ## Description

    Modify a text's style in a text-editor.

    ## Example utterances

    * Using the text editor, give everything the style bold and press Submit.
    * Using the text editor, give everything the style italics and press Submit.
    * Using the text editor, give everything the style underlined and press Submit.
    * Using the text editor, give the text magna. the style bold and press Submit.
    * Using the text editor, give the text sed the color yellow.

    ## Utterance fields

    * target
    """

    subdomain = "text-editor"


class TextTransformEnv(MiniWoBEnvironment):
    """
    ## Description

    Enter slightly transformed text into a text box.

    ## Example utterances

    * Type the text below into the text field and press Submit.

    ## Utterance fields

    (none)
    """

    subdomain = "text-transform"


class TicTacToeEnv(MiniWoBEnvironment):
    """
    ## Description

    Win a game of tic-tac-toe.

    ## Example utterances

    * Playing as 'X', win a game of tic-tac-toe.

    ## Utterance fields

    (none)

    ## Additional notes

    * **Partial reward:** The partial reward is -0.5 for a draw and -0.75 for a loss (not finishing the game gives -1).
    """

    subdomain = "tic-tac-toe"


class UnicodeTestEnv(MiniWoBEnvironment):
    """
    ## Description

    Click on the button with the correct Unicode text.

    ## Example utterances

    * Click on the "Cancél" button.
    * Click on the "ÖK" button.
    * Click on the "ヘルプ" button.
    * Click on the "取消" button.
    * Click on the "确定" button.

    ## Utterance fields

    * target
    """

    subdomain = "unicode-test"


class UseAutocompleteEnv(MiniWoBEnvironment):
    """
    ## Description

    Use autocomplete element efficiently.

    ## Example utterances

    * Enter an item that starts with "An" and ends with "ica".
    * Enter an item that starts with "Ce" and ends with "lic".
    * Enter an item that starts with "Gua" and ends with "la".
    * Enter an item that starts with "Guer".
    * Enter an item that starts with "Rus".

    ## Utterance fields

    * end
    * start
    """

    subdomain = "use-autocomplete"


class UseAutocompleteNodelayEnv(MiniWoBEnvironment):
    """
    ## Description

    [use-autocomplete] Removed delay.

    ## Example utterances

    * Enter an item that starts with "An" and ends with "ica".
    * Enter an item that starts with "Ce" and ends with "lic".
    * Enter an item that starts with "Gua" and ends with "la".
    * Enter an item that starts with "Guer".
    * Enter an item that starts with "Rus".

    ## Utterance fields

    * end
    * start
    """

    subdomain = "use-autocomplete-nodelay"


class UseColorwheelEnv(MiniWoBEnvironment):
    """
    ## Description

    Use a color wheel.

    ## Example utterances

    * Select black with the color picker and hit Submit.
    * Select blue with the color picker and hit Submit.
    * Select gold with the color picker and hit Submit.
    * Select maroon with the color picker and hit Submit.
    * Select yellow with the color picker and hit Submit.

    ## Utterance fields

    * target

    ## Additional notes

    * **Partial reward:** The reward is how far the chosen color is from the specified color.
    * Since the reward is a continuous value, use the `get_thresholded_reward` reward preprocessor to binarize the reward instead of `get_binary_reward`.
    """

    subdomain = "use-colorwheel"


class UseColorwheel2Env(MiniWoBEnvironment):
    """
    ## Description

    Use a color wheel given specific random color.

    ## Example utterances

    * Select the following color with the color picker and hit Submit.

    ## Utterance fields

    (none)

    ## Additional notes

    * **Partial reward:** The reward is how far the chosen color is from the specified color.
    * Since the reward is a continuous value, use the `get_thresholded_reward` reward preprocessor to binarize the reward instead of `get_binary_reward`.
    """

    subdomain = "use-colorwheel-2"


class UseSliderEnv(MiniWoBEnvironment):
    """
    ## Description

    Use a slider to select a particular value.

    ## Example utterances

    * Select -97 with the slider and hit Submit.
    * Select -99 with the slider and hit Submit.
    * Select 0 with the slider and hit Submit.
    * Select 129 with the slider and hit Submit.
    * Select 47 with the slider and hit Submit.

    ## Utterance fields

    * target
    """

    subdomain = "use-slider"


class UseSlider2Env(MiniWoBEnvironment):
    """
    ## Description

    Use sliders to create a given combination.

    ## Example utterances

    * Set the sliders to the combination [11,19,5] and submit.
    * Set the sliders to the combination [12,3,11] and submit.
    * Set the sliders to the combination [17,10,11] and submit.
    * Set the sliders to the combination [19,5,9] and submit.
    * Set the sliders to the combination [6,6,0] and submit.

    ## Utterance fields

    * n1
    * n2
    * n3
    """

    subdomain = "use-slider-2"


class UseSpinnerEnv(MiniWoBEnvironment):
    """
    ## Description

    Use a spinner to select given number.

    ## Example utterances

    * Select -10 with the spinner and hit Submit.
    * Select -3 with the spinner and hit Submit.
    * Select -7 with the spinner and hit Submit.
    * Select 5 with the spinner and hit Submit.
    * Select 7 with the spinner and hit Submit.

    ## Utterance fields

    * target
    """

    subdomain = "use-spinner"


class VisualAdditionEnv(MiniWoBEnvironment):
    """
    ## Description

    Count the total number of blocks.

    ## Example utterances

    * Type the total number of blocks into the textbox and press Submit.

    ## Utterance fields

    (none)
    """

    subdomain = "visual-addition"
