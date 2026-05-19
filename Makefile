# Setup your .venv

PYTHON = .venv/bin/python
FLAKE8 = .venv/bin/flake8

DATA_DIR = data/images
TRANSFORMED_DIR = data/images_transformed
IMG = data/images/Apple_Black_rot/image (1).JPG
IMG_STEM = data/images_transformed/Apple_Black_rot/image (1)
DST = data/images/Apple_Black_rot_test
N = 6
NAME_TAIL = _original
ORIG_TAIL = _original
CHECKPOINT = Model/checkpoints/best.pt
EPOCHS = 5
BATCH_SIZE = 32
ZIP = leaffliction_dataset.zip

all: .venv

.venv:
	@echo "\033[0;32mCreating virtual environment...\033[0m"
	@python3 -m venv .venv
	@.venv/bin/pip install --upgrade pip
	@.venv/bin/pip install -r .libRequirements.txt
	@echo "\033[0;34mTo activate the virtual environment, run 'source .venv/bin/activate'\033[0m"
	@echo "\033[0;34mTo deactivate the virtual environment, run 'deactivate'\033[0m"

fclean:
	@echo -n "\033[0;32mCleaning up...\033[0m"
	@rm -rf .venv
	@find . -type d -name "__pycache__" | xargs rm -rf
	@echo "\033[0;32m\rDone!\033[K\33[0m"

re: fclean all

reset: fclean
	@echo -n "\033[0;32mReset generated files...\033[0m"
	@rm -rf $(TRANSFORMED_DIR) $(CHECKPOINT) $(ZIP) signature.txt
	@echo "\033[0;32m\rDone!\033[K\33[0m"

norm: .venv
	$(FLAKE8) srcs/

distribution: .venv
	$(PYTHON) -m srcs.Distribution.Distribution $(DATA_DIR)

augmentation-one: .venv
	$(PYTHON) -m srcs.Augmentation.Augmentation "$(IMG)"

augmentation: .venv
	$(PYTHON) -m srcs.Augmentation.Augmentation $(DATA_DIR)

transformation-one: .venv
	$(PYTHON) srcs/Transformation/Transformation.py -src "$(IMG)"

transformation: .venv
	$(PYTHON) srcs/Transformation/Transformation.py -src $(DATA_DIR) -dst $(TRANSFORMED_DIR)

train: .venv
	$(PYTHON) -m srcs.Model.train --data_dir $(TRANSFORMED_DIR) --name_tail "$(NAME_TAIL)" --epochs $(EPOCHS) --batch_size $(BATCH_SIZE) --out $(CHECKPOINT)

predict: .venv
	$(PYTHON) -m srcs.Model.predict "$(IMG_STEM)" --name_tail "$(NAME_TAIL)" --orig_tail "$(ORIG_TAIL)" --checkpoint $(CHECKPOINT)

test-imges: .venv
	$(PYTHON) -m srcs.Model.predict "data/test_images/Unit_test1/Apple_scab" --name_tail "" --orig_tail "" --checkpoint $(CHECKPOINT)
	$(PYTHON) -m srcs.Model.predict "data/test_images/Unit_test1/Apple_rust" --name_tail "" --orig_tail "" --checkpoint $(CHECKPOINT)
	$(PYTHON) -m srcs.Model.predict "data/test_images/Unit_test1/Apple_healthy1" --name_tail "" --orig_tail "" --checkpoint $(CHECKPOINT)
	$(PYTHON) -m srcs.Model.predict "data/test_images/Unit_test1/Apple_healthy2" --name_tail "" --orig_tail "" --checkpoint $(CHECKPOINT)
	$(PYTHON) -m srcs.Model.predict "data/test_images/Unit_test2/Grape_Esca" --name_tail "" --orig_tail "" --checkpoint $(CHECKPOINT)
	$(PYTHON) -m srcs.Model.predict "data/test_images/Unit_test2/Grape_healthy" --name_tail "" --orig_tail "" --checkpoint $(CHECKPOINT)
	$(PYTHON) -m srcs.Model.predict "data/test_images/Unit_test2/Grape_Black_rot1" --name_tail "" --orig_tail "" --checkpoint $(CHECKPOINT)
	$(PYTHON) -m srcs.Model.predict "data/test_images/Unit_test2/Grape_Black_rot2" --name_tail "" --orig_tail "" --checkpoint $(CHECKPOINT)
	$(PYTHON) -m srcs.Model.predict "data/test_images/Unit_test2/Grape_spot" --name_tail "" --orig_tail "" --checkpoint $(CHECKPOINT)

signature:
	zip -r $(ZIP) $(TRANSFORMED_DIR) $(CHECKPOINT)
	sha1sum $(ZIP) > signature.txt
	cat signature.txt

.PHONY: all fclean re reset norm distribution augmentation-one augmentation transformation-one transformation train predict test-imges signature
