
SUBDIRS = src

# this recipe is for making the default target of the sub-makefiles
# the following should be a sufficiently "random" string to never cause any problems
default_target_maker_crdaeukntmouerdqjbhtrlchtbhrhckmcrhkbbhetkqn:
	@for dir in $(SUBDIRS); do \
		$(MAKE) -C $${dir} --no-print-directory; \
	done

% :
	@for dir in $(SUBDIRS); do \
		$(MAKE) $@ -C $${dir} --no-print-directory; \
	done
